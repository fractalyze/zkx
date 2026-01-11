/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.
   Copyright 2025 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "zkx/mlir_hlo/stablehlo/dialect/StablehloOps.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cstdint>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep
#include "llvm/Support/Regex.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/InliningUtils.h"

#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "zkx/mlir_hlo/stablehlo/dialect/AssemblyFormat.h"
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloBytecode.h"
#include "zkx/mlir_hlo/stablehlo/dialect/TypeInference.h"

// Include order matters
#define GET_TYPEDEF_CLASSES
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloTypeDefs.cpp.inc"
using mlir::hlo::parseDimSizes;
using mlir::hlo::printDimSizes;
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloAttrs.cpp.inc"

namespace mlir::stablehlo {
namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

hlo::HloDialectInterface *getStablehloDialect(MLIRContext *context) {
  StablehloDialect *dialect = context->getLoadedDialect<StablehloDialect>();
  return dialect->getRegisteredInterface<hlo::HloDialectInterface>();
}

// Returns a new scalar integer value having type `type`. Here `type` must be
// an integer or index type.
Value maybeCastTo(OpBuilder &b, Location loc, Value value, Type type) {
  if (type == value.getType())
    return value;
  assert(type.isIndex() || value.getType().isIndex());
  return b.create<arith::IndexCastOp>(loc, type, value);
}

} // namespace

LogicalResult TypeExtensionsAttr::verifyEncoding(
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  return hlo::verifyBounds(
      getBounds(), RankedTensorType::get(shape, elementType), emitError);
}

//===----------------------------------------------------------------------===//
// CompatibleOperandsAndResultType
//===----------------------------------------------------------------------===//

// TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
// support quantization or sparsity.
#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                         \
  LogicalResult Op::inferReturnTypeComponents(                                 \
      MLIRContext *context, std::optional<Location> location,                  \
      ValueShapeRange operands, DictionaryAttr attributes,                     \
      OpaqueProperties properties, RegionRange regions,                        \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    return inferReturnTypeComponentsFromOperands(                              \
        context, location, operands, attributes, properties, regions,          \
        inferredReturnShapes);                                                 \
  }

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AndOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ClzOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DivOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MaxOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MinOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MulOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NegOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NotOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(OrOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PopulationCountOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PowOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RemOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftLeftOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightArithmeticOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightLogicalOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SignOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SubtractOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(XorOp)

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SmallVector<Type> inferredReturnTypes;
  if (failed(inferReturnTypes(context, location, operands.getValues(),
                              attributes, properties, regions,
                              inferredReturnTypes)))
    return failure();
  if (inferredReturnTypes.size() != 1)
    return failure();
  auto inferredReturnType = dyn_cast<ShapedType>(inferredReturnTypes[0]);
  if (!inferredReturnType)
    return failure();
  inferredReturnShapes.push_back(inferredReturnType);
  return success();
}

LogicalResult AddOp::verify() {
  return hlo::verifyAddOp(getLoc(), getOperation(), getLhs().getType(),
                          getRhs().getType(), getResult().getType());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  mlir::TensorType type = getType();
  if (isa<IntegerType>(type.getElementType())) {
    setNameFn(getResult(), "c");
  } else {
    setNameFn(getResult(), "cst");
  }
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

// static
// Builds a constant op with the specified attribute `value`.
void ConstantOp::build(OpBuilder & /*builder*/, OperationState &result,
                       Type type, Attribute value) {
  if (isa<BoolAttr, IntegerAttr>(value)) {
    // All ZKX types must be tensor types. In the build() method, we want to
    // provide more flexibility by allowing attributes of scalar types. But we
    // need to wrap it up with ElementsAttr to construct valid ZKX constants.
    auto type =
        RankedTensorType::get(/*shape=*/{}, cast<TypedAttr>(value).getType());
    value = DenseElementsAttr::get(type, value);
  }

  result.types.push_back(type);
  result.addAttribute("value", value);
}

LogicalResult
ConstantOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                             ValueRange operands, DictionaryAttr attributes,
                             OpaqueProperties properties, RegionRange,
                             SmallVectorImpl<Type> &inferredReturnTypes) {
  ConstantOpAdaptor adaptor(operands, attributes, properties);
  return hlo::inferConstantOp(location, adaptor.getValue(),
                              inferredReturnTypes);
}

bool ConstantOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  auto lhsTy = dyn_cast<ShapedType>(l.front());
  auto rhsTy = dyn_cast<ShapedType>(r.front());
  if (!lhsTy || !rhsTy)
    return false;

  if (lhsTy == rhsTy)
    return true;

  Type lhsElementType = getElementTypeOrSelf(lhsTy);
  Type rhsElementType = getElementTypeOrSelf(rhsTy);
  // NOTE(chokobole): This allows us to create constants of prime field from
  // integer constants.
  if (isa<IntegerType>(lhsElementType) &&
      isa<prime_ir::field::PrimeFieldType>(rhsElementType)) {
    return lhsTy.clone(rhsElementType) == rhsTy;
  }
  return false;
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  return hlo::parseConstantOp(parser, result);
}

void ConstantOp::print(::mlir::OpAsmPrinter &p) {
  hlo::printConstantOp(p, getOperation(), getValue());
}

//===----------------------------------------------------------------------===//
// CreateTokenOp
//===----------------------------------------------------------------------===//

LogicalResult CreateTokenOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  return hlo::inferCreateTokenOp(getStablehloDialect(context), location,
                                 inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//

LogicalResult GetDimensionSizeOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  GetDimensionSizeOp::Adaptor adaptor(operands, attributes, properties,
                                      regions);
  return hlo::inferGetDimensionSizeOp(location, adaptor.getOperand().getType(),
                                      adaptor.getDimension(),
                                      inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

LogicalResult IotaOp::verify() {
  return hlo::verifyIotaOp(getLoc(), getIotaDimension(), getResult());
}

//===----------------------------------------------------------------------===//
// DynamicIotaOp
//===----------------------------------------------------------------------===//

namespace {

Value castToIndexTensor(OpBuilder &builder, Location loc, Value shapeOp) {
  ShapedType resultTy = shape::getExtentTensorType(
      builder.getContext(), cast<ShapedType>(shapeOp.getType()).getDimSize(0));
  if (shapeOp.getType() == resultTy)
    return shapeOp; // Nothing to do.
  return builder.create<arith::IndexCastOp>(loc, resultTy, shapeOp);
}

} // namespace

LogicalResult DynamicIotaOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  DynamicIotaOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputShape()));
  return success();
}

LogicalResult DynamicIotaOp::verify() {
  return hlo::verifyDynamicIotaOp(getLoc(), getOutputShape(),
                                  getIotaDimension(), getResult());
}

Speculation::Speculatability DynamicIotaOp::getSpeculatability() {
  // If the output shape operand is constant, each of its dimensions is static.
  // For each dimension in the result type's shape:
  // 1. If it is static, the verifier has already checked that it matches the
  //    corresponding dimension in the output shape operand.
  // 2. Otherwise, it is dynamic, so there cannot be a mismatch.
  // (In fact, the result type's shape can be inferred from the operand.)
  if (matchPattern(getOperand(), m_Constant()))
    return Speculation::Speculatable;

  // The result type's shape is fully dynamic, so there cannot be a mismatch
  // with the output shape operand at runtime (the type has no expectations).
  if (llvm::all_of(llvm::seq(getType().getRank()),
                   [this](int64_t i) { return getType().isDynamicDim(i); }))
    return Speculation::Speculatable;

  // The output shape operand's value is unknown and at least one of the result
  // type's dimensions is static, so the dimensions could disagree at runtime.
  return Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// DynamicUpdateSliceOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicUpdateSliceOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  DynamicUpdateSliceOp::Adaptor adaptor(operands, attributes, properties,
                                        regions);
  return hlo::inferDynamicUpdateSliceOp(
      location, adaptor.getOperand(), adaptor.getUpdate(),
      adaptor.getStartIndices(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult
AbsOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  AbsOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferAbsOp(location, adaptor.getOperand(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

// static
void ConvertOp::build(OpBuilder &builder, OperationState &result, Value operand,
                      Type resultElementTy) {
  auto rankedTy = cast<RankedTensorType>(operand.getType());
  auto resultTy = RankedTensorType::get(rankedTy.getShape(), resultElementTy);
  build(builder, result, resultTy, operand);
}

//===----------------------------------------------------------------------===//
// BitcastConvertOp
//===----------------------------------------------------------------------===//

LogicalResult BitcastConvertOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  auto operandType = cast<RankedTensorType>(operands[0].getType());
  RankedTensorType resultType = getType();

  // Shape-changing bitcast convert is not implemented.
  // TODO(kramerb): This could be done by adjusting the last dimension.
  DataLayout dataLayout = DataLayout::closest(*this);
  unsigned operandElementSize =
      dataLayout.getTypeSizeInBits(operandType.getElementType());
  unsigned resultElementSize =
      dataLayout.getTypeSizeInBits(resultType.getElementType());
  if (operandElementSize != resultElementSize)
    return failure();

  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

LogicalResult BitcastConvertOp::verify() {
  return hlo::verifyBitcastConvertOp(getLoc(), getOperand(), getResult());
}

Speculation::Speculatability BitcastConvertOp::getSpeculatability() {
  // The logic is the same as for the
  // SpeculatableIfStaticDimInOutputIsStaticInInput trait, except we don't need
  // to check any "extra" dimension that may result from the difference in bit
  // width of the input and result. Indeed, the extra dimension can be deduced
  // from the bit widths.
  RankedTensorType inputType = getOperand().getType();
  RankedTensorType resultType = getType();
  int64_t rank = std::min(inputType.getRank(), resultType.getRank());
  for (size_t i : llvm::seq(rank)) {
    if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(i))
      return Speculation::NotSpeculatable;
  }
  return Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferBroadcastOp(location, adaptor.getOperand(),
                               adaptor.getBroadcastSizes(),
                               inferredReturnShapes);
}

LogicalResult BroadcastOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();

  auto operandType = cast<RankedTensorType>(operand.getType());

  Location loc = getLoc();
  SmallVector<Value, 4> shapeValues;

  // Collect the broadcast sizes.
  for (int64_t size : getBroadcastSizes())
    shapeValues.push_back(builder.create<arith::ConstantIndexOp>(loc, size));

  // Collect the operand sizes.
  for (auto index : llvm::seq<int64_t>(0, operandType.getRank()))
    shapeValues.push_back(
        builder.createOrFold<tensor::DimOp>(loc, operand, index));

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            builder.getIndexType()),
      shapeValues));

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastInDimOp::verify() {
  return hlo::verifyBroadcastInDimOp(getLoc(), getOperand(),
                                     getBroadcastDimensions(), getResult());
}

namespace {

// Creates BroadcastInDimOp.broadcast_dimensions from BroadcastOp using the
// number of broadcast_sizes and the rank of the operand.
SmallVector<int64_t>
getBroadcastDimensionsFromBroadcast(int64_t broadcastSizesSize,
                                    int64_t operandRank) {
  return llvm::to_vector(
      llvm::seq(broadcastSizesSize, broadcastSizesSize + operandRank));
}

} // namespace

DenseI64ArrayAttr
getBroadcastDimensionsFromBroadcastSizes(RankedTensorType resultType,
                                         DenseI64ArrayAttr broadcastSizes) {
  int64_t broadcastSizesSize = broadcastSizes.size();
  int64_t operandRank = resultType.getRank() - broadcastSizesSize;
  return DenseI64ArrayAttr::get(
      resultType.getContext(),
      getBroadcastDimensionsFromBroadcast(broadcastSizesSize, operandRank));
}

namespace {

// Check that broadcast dimensions are suitable for isSimpleBroadcast():
// extending rank is OK for them, but for dims where rank is not extended the
// dim sizes must match.
//
// Two dimensions are compatible when:
// - they are equal, or
// - one of them is 1.
// and here we reject the case when only one of them is 1.
//
// Examples of compatible dimensions for simple broadcast:
// - tensor<3xf32> -> tensor<1x2x3xf32>
// - tensor<1x1xf32> -> tensor<3x1x1xf32>
// - tensor<5x7xf32> -> tensor<1x3x5x7xf32>
// Examples of non-compatible dimensions:
// - tensor<3xf32> -> tensor<3x3xf32>
// - tensor<3xf32> -> tensor<1x3x3xf32>
// - tensor<1x1xf32> -> tensor<3x2x2xf32>
// - tensor<3x1x1xf32> -> tensor<1x3x5x7xf32>
// - tensor<1x5x7xf32> -> tensor<1x3x5x7xf32>
bool haveSimpleCompatibleDimensions(RankedTensorType operand,
                                    RankedTensorType result) {
  auto operandTy = cast<ShapedType>(operand);
  auto resultTy = cast<ShapedType>(result);
  ArrayRef<int64_t> operandShape = operandTy.getShape();
  ArrayRef<int64_t> resultShape = resultTy.getShape();
  bool isCompatible = true;
  for (auto [operandDim, resultDim] : llvm::zip(operandShape, resultShape))
    isCompatible &= operandDim == resultDim;
  return isCompatible;
}

} // namespace

bool BroadcastInDimOp::isSimpleBroadcast() {
  RankedTensorType operandTy = getOperand().getType();
  RankedTensorType resultTy = getType();
  int64_t operandRank = operandTy.getRank();
  int64_t broadcastSizesSize = resultTy.getRank() - operandRank;
  bool haveCompatibleDimensions =
      haveSimpleCompatibleDimensions(operandTy, resultTy);
  return haveCompatibleDimensions &&
         llvm::to_vector(getBroadcastDimensions()) ==
             getBroadcastDimensionsFromBroadcast(broadcastSizesSize,
                                                 operandRank);
}

//===----------------------------------------------------------------------===//
// DynamicBroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicBroadcastInDimOp::verify() {
  return hlo::verifyDynamicBroadcastInDimOp(
      getLoc(), getOperand(), getOutputDimensions(), getBroadcastDimensions(),
      getKnownExpandingDimensions(), getKnownNonexpandingDimensions(),
      getResult());
}

LogicalResult DynamicBroadcastInDimOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  DynamicBroadcastInDimOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputDimensions()));
  return success();
}

Speculation::Speculatability DynamicBroadcastInDimOp::getSpeculatability() {
  RankedTensorType operandType = getOperand().getType();

  // If input is dynamic, the broadcasting rules might be violated at runtime,
  // so not speculatable.
  if (!operandType.hasStaticShape())
    return Speculation::NotSpeculatable;

  // If input is broadcastable (all 1's) and result is fully dynamic,
  // speculatable.
  bool resultDynamic =
      llvm::all_of(llvm::seq(getType().getRank()),
                   [this](int64_t i) { return getType().isDynamicDim(i); });
  if (operandType.getNumElements() == 1 && resultDynamic)
    return Speculation::Speculatable;

  // If shape is known, speculatable.
  if (matchPattern(getOutputDimensions(), m_Constant()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

LogicalResult ClampOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ClampOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferClampOp(location, adaptor.getMin(), adaptor.getOperand(),
                           adaptor.getMax(), inferredReturnShapes);
}

LogicalResult
ClampOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                               SmallVectorImpl<Value> &reifiedReturnShapes) {
  // For `stablehlo.clamp`, the first operand may be a scalar.
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  ConcatenateOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferConcatenateOp(location, adaptor.getInputs().getTypes(),
                                 adaptor.getDimension(), inferredReturnTypes);
}

LogicalResult ConcatenateOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  ConcatenateOp::Adaptor adaptor(operands);
  ValueRange inputs = adaptor.getInputs();

  Location loc = this->getLoc();
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  SmallVector<SmallVector<Value, 4>, 4> allShapeValues;
  for (size_t inputId = 0; inputId < inputs.size(); ++inputId) {
    Value operand = inputs[inputId];
    auto operandType = cast<RankedTensorType>(operand.getType());

    SmallVector<Value, 4> shapeVals;
    for (const auto &element : llvm::enumerate(operandType.getShape())) {
      Value valueDim = toShapeScalarType(
          builder.create<tensor::DimOp>(loc, operand, element.index()));
      shapeVals.push_back(valueDim);
    }
    allShapeValues.emplace_back(std::move(shapeVals));
  }

  int axis = this->getDimension();
  SmallVector<Value, 4> &shapeValues = allShapeValues[0];
  for (size_t vecId = 1; vecId < allShapeValues.size(); ++vecId) {
    SmallVector<Value, 4> &otherShapeValues = allShapeValues[vecId];
    if (otherShapeValues.size() != shapeValues.size()) {
      this->emitOpError()
          << "Concatenate expects all operands must be of the same rank";
      return failure();
    }
    shapeValues[axis] = builder.create<arith::AddIOp>(loc, shapeValues[axis],
                                                      otherShapeValues[axis]);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

Speculation::Speculatability ConcatenateOp::getSpeculatability() {
  // All operand dimensions must be static, except maybe the concat dim.
  // If concat dim is dynamic, the corresponding dim in operands can be dynamic,
  // otherwise it has to be static.
  int64_t concatDim = getDimension();
  bool concatDimDynamic = getType().isDynamicDim(concatDim);
  for (auto t : getOperandTypes()) {
    auto rankedT = cast<RankedTensorType>(t);
    for (uint64_t i : llvm::seq(rankedT.getRank())) {
      if (i == concatDim && concatDimDynamic)
        continue;
      if (rankedT.isDynamicDim(i))
        return Speculation::NotSpeculatable;
    }
  }
  return Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// DynamicReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicReshapeOp::verify() {
  return hlo::verifyDynamicReshapeOp(getLoc(), getOperand(), getOutputShape(),
                                     getResult());
}

LogicalResult DynamicReshapeOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  DynamicReshapeOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputShape()));
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicSliceOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicSliceOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  DynamicSliceOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferDynamicSliceOp(location, adaptor.getOperand().getType(),
                                  adaptor.getStartIndices().getTypes(),
                                  adaptor.getSliceSizes(),
                                  inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// RealDynamicSliceOp
//===----------------------------------------------------------------------===//

LogicalResult RealDynamicSliceOp::verify() {
  return hlo::verifyRealDynamicSliceOp(getLoc(), getOperand(),
                                       getStartIndices(), getLimitIndices(),
                                       getStrides());
}

LogicalResult RealDynamicSliceOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  RealDynamicSliceOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();
  Value startIndices = adaptor.getStartIndices();
  Value limitIndices = adaptor.getLimitIndices();
  Value strides = adaptor.getStrides();

  auto operandType = cast<RankedTensorType>(operand.getType());

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      cast<ShapedType>(startIndices.getType()).getElementType();
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  one = maybeCastTo(builder, loc, one, shapeScalarType);
  for (const auto &element : llvm::enumerate(operandType.getShape())) {
    Value offset = builder.create<arith::ConstantIndexOp>(loc, element.index());
    Value valueStart =
        builder.create<tensor::ExtractOp>(loc, startIndices, offset);
    Value valueLimit =
        builder.create<tensor::ExtractOp>(loc, limitIndices, offset);
    Value valueStride = builder.create<tensor::ExtractOp>(loc, strides, offset);
    // size = (limit - start + stride - 1) / stride
    shapeValues.push_back(builder.create<arith::DivSIOp>(
        loc,
        builder.create<arith::SubIOp>(
            loc,
            builder.create<arith::AddIOp>(
                loc, valueStride,
                builder.create<arith::SubIOp>(loc, valueLimit, valueStart)),
            one),
        valueStride));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues));
  return success();
}

Speculation::Speculatability RealDynamicSliceOp::getSpeculatability() {
  return hlo::getShapedSpeculatability(getOperation(), /*shapeCount=*/3);
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

LogicalResult MapOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  MapOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferMapOp(location, adaptor.getInputs(), adaptor.getDimensions(),
                         adaptor.getComputation(), inferredReturnShapes);
}

LogicalResult
MapOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                             SmallVectorImpl<Value> &reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

Speculation::Speculatability MapOp::getSpeculatability() {
  // If any dimension of any operand is dynamic, it could disagree with the
  // others at runtime, so the op is not speculatable. If all the operands are
  // statically shaped, whether the op is speculatable or not depends on what
  // ops are in the op's body.
  return llvm::all_of(
             this->getOperation()->getOperandTypes(),
             [](Type t) { return cast<RankedTensorType>(t).hasStaticShape(); })
             ? Speculation::RecursivelySpeculatable
             : Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

void ReduceOp::print(OpAsmPrinter &p) {
  hlo::printReduceOp(p, getOperation(), getInputs(), getDimensions(),
                     getBody());
}

ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &result) {
  auto parseDenseArray = [](OpBuilder &b, ArrayRef<int64_t> dims) -> Attribute {
    return b.getDenseI64ArrayAttr(dims);
  };
  return hlo::parseReduceOp(parser, result, parseDenseArray);
}

LogicalResult ReduceOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ReduceOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferReduceOp(location, adaptor.getInputs().getTypes(),
                            adaptor.getDimensions(), adaptor.getBody(),
                            inferredReturnShapes);
}

// static
void ReduceOp::build(OpBuilder &, OperationState &odsState, ValueRange inputs,
                     ValueRange initValues, DenseI64ArrayAttr dimensions,
                     TypeRange elementTypes) {
  odsState.addOperands(inputs);
  odsState.addOperands(initValues);
  odsState.addAttribute(getDimensionsAttrName(odsState.name), dimensions);
  std::ignore = odsState.addRegion();

  SmallVector<int64_t> newDimensions;
  Attribute encoding;
  ReduceOp::Adaptor adaptor(
      odsState.operands,
      odsState.attributes.getDictionary(odsState.getContext()), {},
      odsState.regions);

  auto inputArgTensorTypes =
      llvm::map_to_vector(adaptor.getInputs().getTypes(),
                          [](Type t) { return cast<ShapedType>(t); });
  auto initValueTensorTypes =
      llvm::map_to_vector(adaptor.getInitValues().getTypes(),
                          [](Type t) { return cast<ShapedType>(t); });

  if (failed(hlo::verifyReduceOpInputsAndInferShape(
          odsState.location, inputArgTensorTypes, dimensions, newDimensions,
          encoding)))
    llvm::report_fatal_error("Failed to infer result type(s).");

  SmallVector<Type> inferredReturnTypes;
  for (auto [inputTy, elementTy] :
       llvm::zip(inputArgTensorTypes, elementTypes)) {
    inferredReturnTypes.push_back(
        RankedTensorType::get(newDimensions, elementTy, encoding));
  }
  odsState.addTypes(inferredReturnTypes);
}

LogicalResult ReduceOp::verify() {
  return hlo::verifyReduceOp(getLoc(), getInputs(), getInitValues(),
                             getDimensions(), getBody());
}

LogicalResult
ReduceOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                SmallVectorImpl<Value> &reifiedReturnShapes) {
  ReduceOp::Adaptor adaptor(operands);
  ValueRange inputs = adaptor.getInputs();

  auto operandType = cast<RankedTensorType>(inputs[0].getType());

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  SmallVector<int64_t, 4> dimensions(this->getDimensions());
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto &element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    auto *it = std::find(dimensions.begin(), dimensions.end(), idx);
    if (it != dimensions.end())
      continue;
    Value valueDim = toShapeScalarType(
        builder.create<tensor::DimOp>(loc, inputs[0], element.index()));
    shapeValues.push_back(valueDim);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  for (size_t i = 0; i < inputs.size(); ++i)
    reifiedReturnShapes.push_back(outputShape);

  return success();
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

LogicalResult ReverseOp::verify() {
  return hlo::verifyReverseOp(getLoc(), getOperand(), getDimensions());
}

LogicalResult
ReverseOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                            ValueRange operands, DictionaryAttr attributes,
                            OpaqueProperties properties, RegionRange regions,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  ReverseOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferReverseOp(location, adaptor.getOperand().getType(),
                             inferredReturnTypes);
}

LogicalResult
ReverseOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                 SmallVectorImpl<Value> &reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SelectOp::Adaptor op(operands, attributes);
  return hlo::inferSelectOp(location, op.getPred(), op.getOnTrue(),
                            op.getOnFalse(), inferredReturnShapes);
}

LogicalResult
SelectOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                SmallVectorImpl<Value> &reifiedReturnShapes) {
  // For `hlo.select`, the first operand may be a scalar.
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// SetDimensionSizeOp
//===----------------------------------------------------------------------===//

LogicalResult SetDimensionSizeOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SetDimensionSizeOp::Adaptor adaptor(operands, attributes, properties,
                                      regions);
  return hlo::inferSetDimensionSizeOp(
      getStablehloDialect(context), location, adaptor.getOperand().getType(),
      adaptor.getSize(), adaptor.getDimension(), inferredReturnShapes);
}

Speculation::Speculatability SetDimensionSizeOp::getSpeculatability() {
  // If the dimension being set is not constant, it is only speculatable if it
  // is dynamic in the output.
  RankedTensorType resultType = getType();
  if (!matchPattern(getSize(), m_Constant()) &&
      !resultType.isDynamicDim(getDimension()))
    return Speculation::NotSpeculatable;

  // For all other dimensions, if the dimension is static in the output, it must
  // be static in the input.
  RankedTensorType inputType = getOperand().getType();
  for (size_t i : llvm::seq(resultType.getRank())) {
    if (i == getDimension())
      continue;
    if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(i))
      return Speculation::NotSpeculatable;
  }
  return Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

// TODO(chokobole): Do we need this? Dependency: interior_padding
LogicalResult
PadOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  PadOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferPadOp(location, adaptor.getOperand().getType(),
                         adaptor.getPaddingValue().getType(),
                         adaptor.getEdgePaddingLow(),
                         adaptor.getEdgePaddingHigh(), inferredReturnTypes);
}

// TODO(chokobole): Do we need this? Dependency: interior_padding
LogicalResult
PadOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                             SmallVectorImpl<Value> &reifiedReturnShapes) {
  PadOp::Adaptor adaptor(operands, getOperation()->getAttrDictionary(),
                         getProperties());
  Location loc = this->getLoc();
  Value operand = adaptor.getOperand();
  auto operandTy = cast<RankedTensorType>(operand.getType());

  ArrayRef<int64_t> padHigh = adaptor.getEdgePaddingHigh();
  ArrayRef<int64_t> padLow = adaptor.getEdgePaddingLow();

  llvm::SmallVector<Value> dimensions;
  dimensions.reserve(operandTy.getRank());
  for (int i = 0, s = operandTy.getRank(); i < s; ++i) {
    Value padEdge =
        builder.create<arith::ConstantIndexOp>(loc, padHigh[i] + padLow[i]);

    // First we grab the initial interior size.
    Value dim = builder.create<tensor::DimOp>(loc, operand, i).getResult();

    // Then we add the padding on the edge of the tensor.
    dim = builder.create<arith::AddIOp>(loc, dim, padEdge).getResult();
    dimensions.push_back(dim);
  }

  Value dimensionTensor =
      builder.create<tensor::FromElementsOp>(loc, dimensions).getResult();
  reifiedReturnShapes.push_back(dimensionTensor);
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicPadOp
//===----------------------------------------------------------------------===//

// TODO(chokobole): Do we need this? Dependency: interior_padding
LogicalResult DynamicPadOp::verify() {
  return hlo::verifyDynamicPadOp(getLoc(), getOperand(), getPaddingValue(),
                                 getEdgePaddingLow(), getEdgePaddingHigh(),
                                 getResult());
}

// TODO(chokobole): Do we need this? Dependency: interior_padding
LogicalResult DynamicPadOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  DynamicPadOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();
  Value edgePaddingLow = adaptor.getEdgePaddingLow();
  Value edgePaddingHigh = adaptor.getEdgePaddingHigh();

  auto operandType = cast<RankedTensorType>(operand.getType());

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      cast<ShapedType>(edgePaddingLow.getType()).getElementType();

  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (int idx : llvm::seq<int>(0, operandType.getShape().size())) {
    Value valueDim =
        toShapeScalarType(builder.create<tensor::DimOp>(loc, operand, idx));
    Value offset = builder.create<arith::ConstantIndexOp>(loc, idx);
    Value valueLow =
        builder.create<tensor::ExtractOp>(loc, edgePaddingLow, offset);
    Value valueHigh =
        builder.create<tensor::ExtractOp>(loc, edgePaddingHigh, offset);
    // output_size = input_size + padding_low + padding_high
    shapeValues.push_back(builder.create<arith::AddIOp>(
        loc, builder.create<arith::AddIOp>(loc, valueDim, valueLow),
        valueHigh));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues));

  return success();
}

Speculation::Speculatability DynamicPadOp::getSpeculatability() {
  return hlo::getShapedSpeculatability(getOperation(), /*shapeCount=*/3);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  return hlo::verifyReshapeOp(getLoc(), getOperand(), getResult());
}

Speculation::Speculatability ReshapeOp::getSpeculatability() {
  if (getOperand().getType().hasStaticShape())
    return Speculation::Speculatable;
  return Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// If Op
//===----------------------------------------------------------------------===//

LogicalResult
IfOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                       ValueRange operands, DictionaryAttr attributes,
                       OpaqueProperties properties, RegionRange regions,
                       SmallVectorImpl<Type> &inferredReturnTypes) {
  IfOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferIfOp(location, adaptor.getPred(), adaptor.getRegions(),
                        inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// Case Op
//===----------------------------------------------------------------------===//

LogicalResult
CaseOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  CaseOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferCaseOp(location, adaptor.getIndex(), adaptor.getRegions(),
                          inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  SliceOpAdaptor adaptor(operands, attributes, properties);
  return hlo::inferSliceOp(location, adaptor.getOperand().getType(),
                           adaptor.getStartIndices(), adaptor.getLimitIndices(),
                           adaptor.getStrides(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

// static
void SortOp::build(OpBuilder &builder, OperationState &state,
                   ValueRange operands, int64_t dimension, bool isStable) {
  state.addOperands(operands);
  state.addAttribute("dimension", builder.getI64IntegerAttr(dimension));
  state.addAttribute("is_stable", builder.getBoolAttr(isStable));

  for (Value operand : operands)
    state.addTypes(operand.getType());

  state.addRegion();
}

LogicalResult SortOp::inferReturnTypeComponents(
    MLIRContext *, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  SortOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferSortOp(location, adaptor.getInputs(), inferredReturnShapes);
}

LogicalResult SortOp::verify() {
  return hlo::verifySortOp(getLoc(), getInputs(), getDimension(),
                           getComparator());
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::reifyReturnTypeShapes(
    OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<Value> &reifiedReturnShapes) {
  TransposeOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();

  auto operandType = cast<RankedTensorType>(operand.getType());

  Location loc = this->getLoc();
  SmallVector<int64_t, 4> permutation(this->getPermutation());
  SmallVector<Value, 4> shapeValues(permutation.size());

  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto &element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    int64_t *it = std::find(permutation.begin(), permutation.end(), idx);
    Value valueDim = toShapeScalarType(
        builder.createOrFold<tensor::DimOp>(loc, operand, element.index()));
    shapeValues[std::distance(permutation.begin(), it)] = valueDim;
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

LogicalResult
TransposeOp::inferReturnTypes(MLIRContext *, std::optional<Location> loc,
                              ValueRange operands, DictionaryAttr attributes,
                              OpaqueProperties properties, RegionRange regions,
                              SmallVectorImpl<Type> &inferredReturnTypes) {
  TransposeOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferTransposeOp(loc, adaptor.getOperand(),
                               adaptor.getPermutation(), inferredReturnTypes);
}

Speculation::Speculatability TransposeOp::getSpeculatability() {
  // This is the same logic as SpeculatableIfStaticDimInOutputIsStaticInInput,
  // except it accounts for the permutation.
  RankedTensorType inputType = getOperand().getType();
  RankedTensorType resultType = getType();
  ArrayRef<int64_t> perm = getPermutation();
  for (size_t i : llvm::seq(resultType.getRank())) {
    if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(perm[i]))
      return Speculation::NotSpeculatable;
  }
  return Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

LogicalResult GetTupleElementOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  GetTupleElementOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferGetTupleElementOp(location, adaptor.getOperand(),
                                     adaptor.getIndex(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

LogicalResult TupleOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  TupleOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferTupleOp(context, location, adaptor.getVal(),
                           inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

LogicalResult CompareOp::inferReturnTypeComponents(
    MLIRContext *context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  CompareOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferCompareOp(context, location, adaptor.getLhs(),
                             inferredReturnShapes);
}

LogicalResult
CompareOp::reifyReturnTypeShapes(OpBuilder &builder, ValueRange operands,
                                 SmallVectorImpl<Value> &reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult
WhileOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                          ValueRange operands, DictionaryAttr attributes,
                          OpaqueProperties properties, RegionRange regions,
                          SmallVectorImpl<Type> &inferredReturnTypes) {
  WhileOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferWhileOp(location, adaptor.getOperand(), inferredReturnTypes);
}

LogicalResult WhileOp::verify() {
  return hlo::verifyWhileOp(getLoc(), getOperand(), getCond(), getBody());
}

void WhileOp::print(OpAsmPrinter &p) {
  hlo::printWhileOp(p, getOperation(), getCond(), getBody());
}

ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  return hlo::parseWhileOp(parser, result);
}

} // namespace mlir::stablehlo

using mlir::hlo::parsePairwiseOpType;
using mlir::hlo::parseSameOperandsAndResultType;
using mlir::hlo::parseSelectOpType;
using mlir::hlo::parseSliceRanges;
using mlir::hlo::parseTupleOpType;
using mlir::hlo::parseVariadicOperandWithAttribute;
using mlir::hlo::parseVariadicSameOperandsAndResultType;
using mlir::hlo::printPairwiseOpType;
using mlir::hlo::printSameOperandsAndResultType;
using mlir::hlo::printSelectOpType;
using mlir::hlo::printSliceRanges;
using mlir::hlo::printTupleOpType;
using mlir::hlo::printVariadicOperandWithAttribute;
using mlir::hlo::printVariadicSameOperandsAndResultType;

#define GET_OP_CLASSES
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloOps.cpp.inc"

namespace mlir::stablehlo {

//===----------------------------------------------------------------------===//
// StableHLO Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct StablehloDialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  // Operations in StableHLO dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};

struct StablehloHloDialectInterface : public hlo::HloDialectInterface {
  using HloDialectInterface::HloDialectInterface;

  Type createTokenType() const override {
    return TokenType::get(getDialect()->getContext());
  }

  bool isTokenType(Type type) const override { return isa<TokenType>(type); }

  Attribute createTypeExtensions(ArrayRef<int64_t> bounds) const override {
    return TypeExtensionsAttr::get(getDialect()->getContext(), bounds);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// StableHLO Dialect Constructor
//===----------------------------------------------------------------------===//

StablehloDialect::StablehloDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<StablehloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloOps.cpp.inc" // NOLINT(build/include)
      >();
  addInterfaces<StablehloDialectInlinerInterface>();
  addInterfaces<StablehloHloDialectInterface>();
  addBytecodeInterface(this);
  addTypes<TokenType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloAttrs.cpp.inc" // NOLINT(build/include)
      >();
}

Type StablehloDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  Type type;
  OptionalParseResult parseResultOpt =
      generatedTypeParser(parser, &mnemonic, type);
  if (parseResultOpt.has_value() && succeeded(*parseResultOpt))
    return type;
  parser.emitError(parser.getNameLoc())
      << "unknown stablehlo type: " << mnemonic;
  return nullptr;
}

void StablehloDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  printer << "<unknown stablehlo type>";
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute StablehloDialect::parseAttribute(DialectAsmParser &parser,
                                           Type type) const {
  StringRef attrTag;
  Attribute attr;
  OptionalParseResult parseResult =
      generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value())
    return attr;
  if (attrTag == "bounds")
    return hlo::parseTypeExtensions(
        parser.getContext()
            ->getOrLoadDialect<StablehloDialect>()
            ->getRegisteredInterface<hlo::HloDialectInterface>(),
        parser);
  parser.emitError(parser.getNameLoc(), "unknown StableHLO attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void StablehloDialect::printAttribute(Attribute attr,
                                      DialectAsmPrinter &os) const {
  if (auto type_extensions = dyn_cast<TypeExtensionsAttr>(attr)) {
    hlo::printTypeExtensions(cast<hlo::BoundedAttrInterface>(attr), os);
    return;
  }
  LogicalResult result = generatedAttributePrinter(attr, os);
  std::ignore = result;
  assert(succeeded(result));
}

namespace {

// Helpers to print an optional array or integer field, to simplify writing
// attribute printers.
template <typename T>
void printField(AsmPrinter &printer, StringRef name, T field,
                StringRef &separator) {
  if (field != 0) {
    printer << separator << name << " = " << field;
    separator = ", ";
  }
}

template <typename T>
void printField(AsmPrinter &printer, StringRef name, ArrayRef<T> field,
                StringRef &separator) {
  if (!field.empty()) {
    printer << separator << name << " = [";
    llvm::interleaveComma(field, printer);
    printer << "]";
    separator = ", ";
  }
}

template <typename... Ts>
void printStruct(AsmPrinter &printer, StringRef name, Ts... printFields) {
  printer << "<";
  StringRef separator = "";
  // Fold expression to print each entry in the parameter pack.
  // TODO(stablehlo-team): this can be simplified when TF moves to C++17.
  using unused = int[];
  std::ignore = unused{0, (printField(printer, std::get<0>(printFields),
                                      std::get<1>(printFields), separator),
                           0)...};
  printer << ">";
}

} // namespace

//===----------------------------------------------------------------------===//
// Builder utilities
//===----------------------------------------------------------------------===//

namespace {

// Builds the region `body` for stablehlo.sort's comparator: for each type in
// `element_types`, create two block arguments, one for lhs and one for rhs, and
// generates stablehlo.compare op to compare them with the given `direction`.
//
// Note that this right now only does comparison on the first pair of block
// arguments.
void buildSortComparisonBody(llvm::ArrayRef<Type> elementTypes,
                             ComparisonDirection direction, Region *body,
                             OpBuilder *builder) {
  OpBuilder::InsertionGuard insertionPointGuard(*builder);

  Location loc = body->getLoc();
  Block *block = builder->createBlock(body);
  // Add two arguments for each element type.
  for (Type elementType : elementTypes) {
    ShapedType shapedType = RankedTensorType::get({}, elementType);
    block->addArguments({shapedType, shapedType},
                        SmallVector<Location, 2>(2, loc));
  }

  Value compare = builder->create<CompareOp>(loc, block->getArgument(0),
                                             block->getArgument(1), direction);

  builder->create<ReturnOp>(loc, compare);
}

} // namespace

SortOp createSortOp(PatternRewriter *rewriter, const Location &loc,
                    const llvm::ArrayRef<Value> &operands,
                    const llvm::ArrayRef<Type> &elementTypes, int64_t dimension,
                    bool isStable, ComparisonDirection direction) {
  assert(!operands.empty() && "No operands to sort");
  // Create the sort op.
  auto sortOp = rewriter->create<SortOp>(loc, operands, dimension, isStable);
  buildSortComparisonBody(elementTypes, direction, &sortOp.getComparator(),
                          rewriter);
  return sortOp;
}

//===----------------------------------------------------------------------===//
// StableHLO Dialect Hooks
//===----------------------------------------------------------------------===//

Operation *StablehloDialect::materializeConstant(OpBuilder &builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
  auto elementsAttr = dyn_cast<ElementsAttr>(value);
  // HLO dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (!elementsAttr)
    return nullptr;
  // HLO dialect constants require the type of value and result to match.
  if (type != elementsAttr.getType())
    return nullptr;

  return builder.create<ConstantOp>(loc, type, elementsAttr);
}

std::optional<StablehloDialectVersion> StablehloDialect::getVersion() const {
  return version;
}

void StablehloDialect::setVersion(
    std::optional<StablehloDialectVersion> version) {
  this->version = version;
}

} // namespace mlir::stablehlo
