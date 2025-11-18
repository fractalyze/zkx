/* Copyright 2019 The OpenXLA Authors.
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

// This file defines the operations used in the MHLO dialect.

#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"

#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h.inc"
#include "zkx/mlir_hlo/stablehlo/dialect/AssemblyFormat.h"
#include "zkx/mlir_hlo/stablehlo/dialect/TypeInference.h"
#include "zkx/mlir_hlo/utils/convert_op_folder.h"

namespace mlir {
#include "zkx/mlir_hlo/mhlo/IR/hlo_patterns.cc.inc"
}  // namespace mlir

using mlir::hlo::parseDimSizes;
using mlir::hlo::printDimSizes;

#include "zkx/mlir_hlo/mhlo/IR/hlo_ops_enums.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops_attrs.cc.inc"
#define GET_TYPEDEF_CLASSES
// TODO(chokobole): Uncomment this. Dependency: hlo_ops_typedefs.td
// #include "zkx/mlir_hlo/mhlo/IR/hlo_ops_typedefs.cc.inc"

namespace mlir::mhlo {
namespace {

//===----------------------------------------------------------------------===//
// Utilities for the canonicalize patterns
//===----------------------------------------------------------------------===//

// This is an upper limit on how many elements can be folded by an op folder.
// This limit doesn't apply to some special cases like adding a zero,
// multiplying by one, doing many operations with splats.
constexpr int64_t kFoldOpEltLimit = 65536;

// Clamps value to the range [lower, upper]. Requires lower <= upper.
template <typename T>
T clamp(const T& value, const T& lower, const T& upper) {
  assert(lower <= upper);
  return std::max(lower, std::min(value, upper));
}

// Verifies that dimension attribute for the op correctly indexes in operand or
// result shape.
template <typename OpT>
LogicalResult verifyDimAttr(OpT op) {
  int64_t rank = -1;
  if (auto ty = dyn_cast<RankedTensorType>(op.getOperand().getType())) {
    rank = ty.getRank();
  } else if (auto ty = dyn_cast<RankedTensorType>(op.getType())) {
    rank = ty.getRank();
  } else {
    return success();
  }

  int64_t dim = op.getDimension();
  if (dim < 0 || dim >= rank)
    return op.emitOpError() << "requires dimension attribute in range [0, "
                            << rank << "); found (" << dim << ")";
  return success();
}

}  // namespace

// TODO(chokobole): Uncomment this. Dependency: mhlo_canonicalize.td
// #include "zkx/mlir_hlo/mhlo/IR/mhlo_canonicalize.inc"

//===----------------------------------------------------------------------===//
// Utilities for verifiers
//===----------------------------------------------------------------------===//

namespace {

LogicalResult verify1dTensor(std::optional<Location> loc,
                             DenseIntElementsAttr attr,
                             std::string_view attrName) {
  int64_t rank = attr.getType().getRank();
  if (rank != 1) {
    return emitOptionalError(loc, attrName, " has rank ", rank,
                             " instead of required rank 1.");
  }
  return success();
}

}  // namespace

LogicalResult TypeExtensionsAttr::verifyEncoding(
    ArrayRef<int64_t> shape, mlir::Type elementType,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  return hlo::verifyBounds(
      getBounds(), RankedTensorType::get(shape, elementType), emitError);
}

//===----------------------------------------------------------------------===//
// CompatibleOperandsAndResultType
//===----------------------------------------------------------------------===//

// TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
// support sparsity.
#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                \
  LogicalResult Op::inferReturnTypeComponents(                        \
      MLIRContext* context, std::optional<Location> location,         \
      ValueShapeRange operands, DictionaryAttr attributes,            \
      OpaqueProperties properties, RegionRange regions,               \
      SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {  \
    return inferReturnTypeComponentsFromOperands(                     \
        context, location, operands, attributes, properties, regions, \
        inferredReturnShapes);                                        \
  }

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AddOp)
// TODO(chokobole): uncomment this. Dependency: AndOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AndOp)
// TODO(chokobole): uncomment this. Dependency: CollectiveBroadcastOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CollectiveBroadcastOp)
// TODO(chokobole): uncomment this. Dependency: CollectivePermuteOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CollectivePermuteOp)
// TODO(chokobole): uncomment this. Dependency: CopyOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CopyOp)
// TODO(chokobole): uncomment this. Dependency: CrossReplicaSumOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CrossReplicaSumOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DivOp)
// TODO(chokobole): uncomment this. Dependency: DomainOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DomainOp)
// TODO(chokobole): uncomment this. Dependency: MaxOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MaxOp)
// TODO(chokobole): uncomment this. Dependency: MinOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MinOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MulOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NegOp)
// TODO(chokobole): uncomment this. Dependency: NotOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NotOp)
// TODO(chokobole): uncomment this. Dependency: OrOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(OrOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PowOp)
// TODO(chokobole): uncomment this. Dependency: ReverseOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReverseOp)
// TODO(chokobole): uncomment this. Dependency: ShiftLeftOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftLeftOp)
// TODO(chokobole): uncomment this. Dependency: ShiftRightArithmeticOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightArithmeticOp)
// TODO(chokobole): uncomment this. Dependency: ShiftRightLogicalOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightLogicalOp)
// TODO(chokobole): uncomment this. Dependency: SignOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SignOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SubtractOp)
// TODO(chokobole): uncomment this. Dependency: XorOp
// INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(XorOp)

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

// static
// Builds a constant op with the specified attribute `value`.
void ConstantOp::build(OpBuilder& /*builder*/, OperationState& result,
                       Attribute value) {
  Properties& properties = result.getOrAddProperties<Properties>();
  Type type;
  if (auto elemAttr = dyn_cast<ElementsAttr>(value)) {
    type = elemAttr.getType();
    properties.value = elemAttr;
  } else if (isa<BoolAttr, IntegerAttr>(value)) {
    // All ZKX types must be tensor types. In the build() method, we want to
    // provide more flexibility by allowing attributes of scalar types. But we
    // need to wrap it up with ElementsAttr to construct valid ZKX constants.
    type =
        RankedTensorType::get(/*shape=*/{}, cast<TypedAttr>(value).getType());
    properties.value = DenseElementsAttr::get(cast<TensorType>(type), value);
  }

  // TODO: support other ZKX specific types.
  assert(type && "unsupported attribute type for building mhlo.constant");
  result.types.push_back(type);
}

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ConstantOpAdaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferConstantOp(location, adaptor.getValue(),
                              inferredReturnTypes);
}

bool ConstantOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1) return false;
  auto lhsTy = cast<ShapedType>(l.front());
  auto rhsTy = cast<ShapedType>(r.front());
  return lhsTy == rhsTy;
}

ParseResult ConstantOp::parse(OpAsmParser& parser, OperationState& result) {
  return hlo::parseConstantOp(parser, result);
}

void ConstantOp::print(OpAsmPrinter& p) {
  hlo::printConstantOp(p, getOperation(), getValue());
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void ConvertOp::build(OpBuilder& builder, OperationState& result, Value operand,
                      Type resultElementTy) {
  auto rankedTy = cast<RankedTensorType>(operand.getType());
  auto resultTy = RankedTensorType::get(rankedTy.getShape(), resultElementTy);
  build(builder, result, resultTy, operand);
}

OpFoldResult ConvertOp::fold(FoldAdaptor adaptor) {
  ArrayRef<Attribute> operands = adaptor.getOperands();
  auto operandTy = cast<TensorType>(getOperand().getType());
  auto resultTy = cast<TensorType>(getResult().getType());
  if (operandTy == resultTy) return getOperand();

  // If the result has non-static shape, a convert op is necessary to go from
  // static shape to non-static shape.
  if (!resultTy.hasStaticShape()) return {};

  // If the operand is constant, we can do the conversion now.
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(operands.front());
  if (!elementsAttr) return {};

  // Prevent folding if the result is too large.
  if (elementsAttr.getNumElements() > kFoldOpEltLimit) return {};
  return hlo::convertElementsAttr(elementsAttr,
                                  getElementTypeOrSelf(getResult()));
}

namespace {

struct EliminateRedundantConvert : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern<ConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter& rewriter) const override {
    auto convertOp = op.getOperand().getDefiningOp<ConvertOp>();
    if (!convertOp) {
      return failure();
    }
    auto firstType =
        cast<TensorType>(convertOp.getOperand().getType()).getElementType();
    auto secondType =
        cast<TensorType>(op.getOperand().getType()).getElementType();
    auto thirdType =
        cast<TensorType>(op.getResult().getType()).getElementType();
    Location loc = rewriter.getFusedLoc({convertOp->getLoc(), op->getLoc()});
    if (isa<IntegerType>(firstType) && isa<IntegerType>(secondType) &&
        isa<IntegerType>(thirdType)) {
      // fold when the second integer type's width is longer than first,
      // like i16 -> i32 -> i64, u16 -> i32 -> u32
      if (cast<IntegerType>(secondType).getWidth() >
          cast<IntegerType>(firstType).getWidth()) {
        Value result = rewriter.create<ConvertOp>(loc, op.getResult().getType(),
                                                  convertOp.getOperand());
        rewriter.replaceOp(op, result);
        return success();
      }
    }
    return failure();
  }
};

}  // namespace

void ConvertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<EliminateIdentityConvert>(context);
  results.add<EliminateRedundantConvert>(context);
}

//===----------------------------------------------------------------------===//
// UnaryOps
//===----------------------------------------------------------------------===//

namespace {

template <typename ValType>
struct AnyValue {
  bool operator()(const ValType&) { return true; }
};

template <typename ValType>
struct NonNegativeValue {
  bool operator()(const ValType& v) { return !v.isNegative(); }
};

template <typename ValType>
struct PositiveValue {
  bool operator()(const ValType& v) { return !v.isNegative() && !v.isZero(); }
};

APSInt addSign(const APInt& v, Type t) {
  // Add signedness information to the value, treating signless as signed,
  // unless it's i1.
  return APSInt(v, t.isUnsignedInteger() || t.isSignlessInteger(1));
}

template <typename Op, typename ElementType, typename ValType, typename Convert,
          typename Validate = AnyValue<ValType>>
Attribute UnaryFolder(Op* op, ArrayRef<Attribute> attrs) {
  if (!attrs[0]) return {};

  auto val = dyn_cast<DenseElementsAttr>(attrs[0]);
  if (!val) return {};

  auto type = cast<ShapedType>(op->getType());
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!isa<ElementType>(etype)) {
    return {};
  }

  // Prevent folding if the result is too large.
  if (val.getNumElements() > kFoldOpEltLimit) return {};

  SmallVector<ValType, 6> values;
  values.reserve(val.getNumElements());
  for (const auto v : val.getValues<ValType>()) {
    if (!Validate()(v)) return {};
    std::optional<ValType> r = Convert()(addSign(v, type));
    if (!r) return {};
    values.push_back(r.value());
  }

  return DenseElementsAttr::get(type, values);
}

}  // namespace

// NOLINTBEGIN(bugprone-macro-parentheses)
#define UNARY_FOLDER(Op, Func)                                              \
  OpFoldResult Op::fold(FoldAdaptor adaptor) {                              \
    auto attrs = adaptor.getOperands();                                     \
    if (isa<IntegerType>(getElementTypeOrSelf(getType())))                  \
      return UnaryFolder<Op, IntegerType, APInt, Func<APInt>>(this, attrs); \
    return {};                                                              \
  }

UNARY_FOLDER(NegOp, std::negate)
// TODO(chokobole): Uncomment this. Dependency: SignOp
// UNARY_FOLDER(SignOp, Sign)
// TODO(chokobole): Uncomment this. Dependency: AbsOp
// UNARY_FOLDER(AbsOp, Abs)
// TODO(chokobole): Uncomment this. Dependency: NotOp
// UNARY_FOLDER(NotOp, std::bit_not)

#undef UNARY_FOLDER

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

namespace {

template <typename Op, typename ElementType = Type, typename ValType,
          typename Convert>
Attribute BinaryFolder(Op* op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1]) return {};

  auto lhs = dyn_cast<DenseElementsAttr>(attrs[0]);
  auto rhs = dyn_cast<DenseElementsAttr>(attrs[1]);
  if (!lhs || !rhs) return {};

  auto type = cast<ShapedType>(op->getType());
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!isa<ElementType>(etype)) {
    return {};
  }

  // Special case for folding splats no matter how large.
  // Only covers the case of both attrs being splats; operation-specific cases
  // like adding a zero or multiplying by one are handled elsewhere.
  auto splatLhs = dyn_cast<SplatElementsAttr>(lhs);
  auto splatRhs = dyn_cast<SplatElementsAttr>(rhs);
  if (splatLhs && splatRhs) {
    APSInt signedLhs = addSign(splatLhs.getSplatValue<ValType>(), etype);
    APSInt signedRhs = addSign(splatRhs.getSplatValue<ValType>(), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    return succeeded(result) ? SplatElementsAttr::get(type, *result)
                             : Attribute();
  }

  // Prevent folding if the result is too large.
  if (lhs.getNumElements() > kFoldOpEltLimit) return {};

  SmallVector<ValType, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<ValType>(), rhs.getValues<ValType>())) {
    APSInt signedLhs = addSign(std::get<0>(zip), etype);
    APSInt signedRhs = addSign(std::get<1>(zip), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    if (failed(result)) {
      return {};
    }
    values.push_back(std::move(*result));
  }

  return DenseElementsAttr::get(type, values);
}

template <typename T>
struct Divide : std::divides<T> {};

template <>
struct Divide<APSInt> {
  FailureOr<APSInt> operator()(const APSInt& a, const APSInt& b) const {
    if (b.isZero()) return failure();
    return a / b;
  }
};

template <typename T>
struct Max {
  T operator()(const T& a, const T& b) const { return std::max<T>(a, b); }
};

template <typename T>
struct Min {
  T operator()(const T& a, const T& b) const { return std::min<T>(a, b); }
};

}  // namespace

#define BINARY_FOLDER_INTERNAL(Op, Func)                                    \
  if (isa<IntegerType>(getElementTypeOrSelf(getType())))                    \
    return BinaryFolder<Op, IntegerType, APInt, Func<APSInt>>(this, attrs); \
  return {};

#define BINARY_FOLDER(Op, Func)                \
  OpFoldResult Op::fold(FoldAdaptor adaptor) { \
    auto attrs = adaptor.getOperands();        \
    BINARY_FOLDER_INTERNAL(Op, Func)           \
  }

// Addition, subtraction and multiplication use the std:: versions of the ops.
// Due to the other ops behaving differently in signed vs unsigned integers,
// APInts need a special implementation. Currently, it replicates signed int
// op behavior.
BINARY_FOLDER(SubtractOp, std::minus)
BINARY_FOLDER(DivOp, Divide)
// TODO(chokobole): Uncomment this. Dependency: Remainder
// BINARY_FOLDER(RemOp, Remainder)
// TODO(chokobole): Uncomment this. Dependency: Max
// BINARY_FOLDER(MaxOp, Max)
// TODO(chokobole): Uncomment this. Dependency: Min
// BINARY_FOLDER(MinOp, Min)

namespace {

bool isSplatZero(SplatElementsAttr attr) {
  if (!attr) return false;
  if (isa<IntegerType>(attr.getElementType())) {
    return attr.getSplatValue<APInt>().isZero();
  }
  return false;
}

}  // namespace

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto attrs = adaptor.getOperands();
  // Handle special case where one operand is 0:  x + 0 => x
  if (attrs[0] || attrs[1]) {
    auto splatLhs = dyn_cast_or_null<SplatElementsAttr>(attrs[0]);
    auto splatRhs = dyn_cast_or_null<SplatElementsAttr>(attrs[1]);
    if (isSplatZero(splatLhs))
      return splatRhs ? static_cast<OpFoldResult>(splatRhs) : getRhs();
    if (isSplatZero(splatRhs))
      return splatLhs ? static_cast<OpFoldResult>(splatLhs) : getLhs();
  }
  if (attrs[0] && attrs[1]) {
    BINARY_FOLDER_INTERNAL(AddOp, std::plus)
  }
  return {};
}

namespace {

bool isSplatOne(SplatElementsAttr attr) {
  if (!attr) return false;
  if (isa<IntegerType>(attr.getElementType())) {
    return attr.getSplatValue<APInt>().getSExtValue() == 1;
  }
  return false;
}

}  // namespace

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto attrs = adaptor.getOperands();
  // Handle special case where one operand is 1: x * 1 => x
  if (attrs[0] || attrs[1]) {
    auto splatLhs = dyn_cast_or_null<SplatElementsAttr>(attrs[0]);
    auto splatRhs = dyn_cast_or_null<SplatElementsAttr>(attrs[1]);
    if (isSplatOne(splatLhs))
      return splatRhs ? static_cast<OpFoldResult>(splatRhs) : getRhs();
    if (isSplatOne(splatRhs))
      return splatLhs ? static_cast<OpFoldResult>(splatLhs) : getLhs();
  }
  if (attrs[0] && attrs[1]) {
    BINARY_FOLDER_INTERNAL(MulOp, std::multiplies);
  }
  return {};
}

#undef BINARY_FOLDER_INTERNAL
#undef BINARY_FOLDER

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext* /*context*/, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
  SliceOpAdaptor adaptor(operands, attributes, properties, regions);
  if (failed(verify1dTensor(location, adaptor.getStartIndices(),
                            "start_indices")) ||
      failed(verify1dTensor(location, adaptor.getLimitIndices(),
                            "limit_indices")) ||
      failed(verify1dTensor(location, adaptor.getStrides(), "strides")))
    return failure();
  return hlo::inferSliceOp(
      location, adaptor.getOperand().getType(),
      llvm::to_vector(adaptor.getStartIndices().getValues<int64_t>()),
      llvm::to_vector(adaptor.getLimitIndices().getValues<int64_t>()),
      llvm::to_vector(adaptor.getStrides().getValues<int64_t>()),
      inferredReturnTypes);
}

namespace {

template <typename I, typename E>
void sliceElements(I values, ArrayRef<int64_t> sizes, ArrayRef<int64_t> starts,
                   ArrayRef<int64_t> limits, ArrayRef<int64_t> strides,
                   llvm::SmallVectorImpl<E>* outValues) {
  assert(starts.size() == limits.size());
  assert(starts.size() == strides.size());
  if (starts.empty()) return;

  int64_t start = starts.front();
  int64_t limit = limits.front();
  int64_t stride = strides.front();
  if (starts.size() == 1) {
    for (int i = start; i < limit; i += stride) {
      outValues->push_back(*(values + i));
    }
    return;
  }

  for (; start < limit; start += stride) {
    auto begin = values + start * sizes.front();
    sliceElements<I, E>(begin, sizes.drop_front(), starts.drop_front(),
                        limits.drop_front(), strides.drop_front(), outValues);
  }
}

template <typename I, typename E>
Attribute foldSlice(SliceOp* op, I values) {
  auto start = llvm::to_vector<6>(op->getStartIndices().getValues<int64_t>());
  auto limit = llvm::to_vector<6>(op->getLimitIndices().getValues<int64_t>());
  auto stride = llvm::to_vector<6>(op->getStrides().getValues<int64_t>());

  // TODO(b/235903849): This should be op->getType().case<ShapedType>().
  auto resultType = cast<ShapedType>(op->getOperand().getType());
  if (!resultType.hasStaticShape()) return {};

  ArrayRef<int64_t> shape = resultType.getShape();
  int64_t count = resultType.getNumElements();
  if (count == 0) {
    return DenseElementsAttr::get<E>(
        cast<ShapedType>(op->getResult().getType()),
        /*list=*/{});
  }

  // Compute the striding for each dimension.
  llvm::SmallVector<int64_t, 6> sizes;
  sizes.reserve(shape.size());
  for (auto v : shape) {
    count = count / v;
    sizes.push_back(count);
  }

  // Prevent folding if the result is too large.
  if (resultType.getNumElements() > kFoldOpEltLimit) return {};

  llvm::SmallVector<E, 6> outValues;
  outValues.reserve(resultType.getNumElements());
  sliceElements<I, E>(values, sizes, start, limit, stride, &outValues);

  return DenseElementsAttr::get(cast<ShapedType>(op->getResult().getType()),
                                outValues);
}

}  // namespace

OpFoldResult SliceOp::fold(FoldAdaptor adaptor) {
  ArrayRef<Attribute> operands = adaptor.getOperands();
  // Check if the SliceOp is a NoOp operation.
  auto operandType = cast<ShapedType>(getOperand().getType());
  auto resultType = cast<ShapedType>(getResult().getType());

  if (operandType.hasStaticShape() && resultType.hasStaticShape() &&
      (operandType.getShape() == resultType.getShape())) {
    return getOperand();
  }

  if (operands.empty() || !operands.front()) return {};

  // Evaluate for statically valued inputs.
  auto elements = dyn_cast<DenseElementsAttr>(operands.front());
  if (!elements) return {};

  auto etype = elements.getType().getElementType();
  if (isa<IntegerType>(etype)) {
    return foldSlice<DenseElementsAttr::IntElementIterator, APInt>(
        this, elements.value_begin<APInt>());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

namespace {

// Pattern for unpacking and repacking the same tuple.
struct UnpackRepackSameTuple : public OpRewritePattern<TupleOp> {
  using OpRewritePattern<TupleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TupleOp op,
                                PatternRewriter& rewriter) const override {
    // TODO(chokobole): Uncomment this. Dependency: GetTupleElementOp
    // if (op.getVal().empty()) return failure();

    // Value firstElement = op.getVal().front();
    // auto firstElementOp = firstElement.getDefiningOp<GetTupleElementOp>();
    // if (!firstElementOp || firstElementOp.getIndexAttr().getInt() != 0)
    //   return failure();

    // Value tuplePredecessor = firstElementOp.getOperand();
    // if (tuplePredecessor.getType() != op.getType()) return failure();

    // for (const auto& elementAndIdx :
    //      llvm::enumerate(op.getVal().drop_front(1))) {
    //   auto elementOp =
    //   elementAndIdx.value().getDefiningOp<GetTupleElementOp>(); if
    //   (!elementOp ||
    //       elementOp.getIndexAttr().getInt() !=
    //           static_cast<int64_t>(elementAndIdx.index() + 1) ||
    //       elementOp.getOperand() != tuplePredecessor)
    //     return failure();
    // }

    // rewriter.replaceOp(op, tuplePredecessor);
    // return success();
    return failure();
  }
};

}  // namespace

void TupleOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<UnpackRepackSameTuple>(context);
}

LogicalResult TupleOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  TupleOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferTupleOp(context, location, adaptor.getVal(),
                           inferredReturnTypes);
}

}  // namespace mlir::mhlo

using mlir::hlo::parseSameOperandsAndResultType;
using mlir::hlo::parseTupleOpType;
using mlir::hlo::printSameOperandsAndResultType;
using mlir::hlo::printTupleOpType;

#define GET_OP_CLASSES
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.cc.inc"

namespace mlir::mhlo {

//===----------------------------------------------------------------------===//
// mhlo Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct MhloDialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       IRMapping& valueMapping) const final {
    return true;
  }
  // Operations in mhlo dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final {
    return true;
  }
};

struct MhloHloDialectInterface : public hlo::HloDialectInterface {
  using HloDialectInterface::HloDialectInterface;

  Type createTokenType() const override {
    return TokenType::get(getDialect()->getContext());
  }

  bool isTokenType(Type type) const override { return isa<TokenType>(type); }

  Attribute createTypeExtensions(ArrayRef<int64_t> bounds) const override {
    return TypeExtensionsAttr::get(getDialect()->getContext(), bounds);
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// mhlo Dialect Constructor
//===----------------------------------------------------------------------===//

MhloDialect::MhloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<MhloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.cc.inc"  // NOLINT(build/include)
      >();
  addInterfaces<MhloHloDialectInterface>();
  addInterfaces<MhloDialectInlinerInterface>();
  // TODO(chokobole): Uncomment this. Dependency: MhloBytecodeInterface
  // addBytecodeInterface(this);
  // TODO(chokobole): Uncomment this. Dependency: AsyncBundleType
  // addTypes<TokenType, AsyncBundleType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops_attrs.cc.inc"  // NOLINT(build/include)
      >();
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute MhloDialect::parseAttribute(DialectAsmParser& parser,
                                      Type type) const {
  StringRef attrTag;
  Attribute attr;
  OptionalParseResult parseResult =
      generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value()) return attr;
  parser.emitError(parser.getNameLoc(), "unknown mhlo attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void MhloDialect::printAttribute(Attribute attr, DialectAsmPrinter& os) const {
  LogicalResult result = generatedAttributePrinter(attr, os);
  std::ignore = result;
  assert(succeeded(result));
}

//===----------------------------------------------------------------------===//
// MHLO Dialect Hooks
//===----------------------------------------------------------------------===//

Operation* MhloDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                            Type type, Location loc) {
  auto elementsAttr = dyn_cast<ElementsAttr>(value);
  // HLO dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (!elementsAttr) return nullptr;
  auto resultShapedType = dyn_cast<ShapedType>(type);
  auto attrShapedType = dyn_cast<ShapedType>(elementsAttr.getType());
  if (resultShapedType && attrShapedType) {
    return builder.create<mhlo::ConstantOp>(loc, type, elementsAttr);
  }
  // HLO dialect constants require the type of value and result to match
  if (type != elementsAttr.getType()) return nullptr;

  return builder.create<mhlo::ConstantOp>(loc, type, elementsAttr);
}

LogicalResult MhloDialect::verifyRegionArgAttribute(Operation* op,
                                                    unsigned /*regionIndex*/,
                                                    unsigned argIndex,
                                                    NamedAttribute attr) {
  // TODO(chokobole): Uncomment this. Dependency: ArgResultAliasAttr
  // if (auto aliasAttr = dyn_cast<ArgResultAliasAttr>(attr.getValue())) {
  //   if (failed(
  //           verifyArgResultAliasAttr(attr.getName(), aliasAttr, argIndex,
  //           op)))
  //     return failure();
  // }
  // TODO(chokobole): Uncomment this. Dependency: ParameterReplicationAttr
  // if (attr.getName() == "mhlo.parameter_replication") {
  //   auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue());
  //   if (!arrayAttr)
  //     return op->emitOpError() << "parameter_replication: must be an array";
  //   auto func = dyn_cast<FunctionOpInterface>(op);
  //   if (!func) {
  //     return op->emitOpError()
  //            << "has parameter_replication but is not a function";
  //   }
  //   // parameter_replication = [] or [false] is equivalent to
  //   // [false,...,false] and parameter_replication = [true] means
  //   // [true,...,true]
  //   if (arrayAttr.empty() || arrayAttr.size() == 1) return success();
  //   auto num_leaf_buffers =
  //       getNumLeafBuffers(func.getArgumentTypes()[argIndex]);
  //   if ((size_t)num_leaf_buffers != arrayAttr.size())
  //     return op->emitOpError()
  //            << "parameter_replication: arg " << argIndex << " has "
  //            << num_leaf_buffers << " leaf_buffers, but
  //            parameter_replication"
  //            << " expects " << arrayAttr.size();
  // }
  return success();
}

LogicalResult MhloDialect::verifyOperationAttribute(Operation* op,
                                                    NamedAttribute attr) {
  // TODO(chokobole): Uncomment this. Dependency: ArgResultAliasAttr
  // if (auto aliasAttr = dyn_cast<ArgResultAliasAttr>(attr.getValue())) {
  //   if (!isa<FunctionOpInterface>(op))
  //     return op->emitOpError()
  //            << "attribute " << attr.getName()
  //            << " can only be used on function-like operations";
  // }
  // TODO(chokobole): Uncomment this. Dependency: CrossProgramPrefetchAttr
  // if (attr.getName() == "mhlo.cross_program_prefetches") {
  //   auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue());
  //   if (!arrayAttr)
  //     return op->emitOpError() << "cross_program_prefetches must be an
  //     array";
  //   for (auto attrElt : arrayAttr) {
  //     auto prefetchAttr = dyn_cast<CrossProgramPrefetchAttr>(attrElt);
  //     if (!prefetchAttr)
  //       return op->emitOpError() << "cross_program_prefetches must be an
  //       array "
  //                                   "of cross_program_prefetch attrs";
  //     auto module = dyn_cast<ModuleOp>(op);
  //     if (!module)
  //       return op->emitOpError()
  //              << "has cross_program_prefetches but is not a module";
  //     auto res = verifyCrossProgramPrefetchAttr(prefetchAttr, module);
  //     if (failed(res)) return res;
  //   }
  // }
  if (attr.getName() == "mhlo.spmd_parameters_sharding") {
    auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue());
    if (!arrayAttr)
      return op->emitOpError() << "spmd_parameters_sharding: must be an array";
    auto module = dyn_cast<ModuleOp>(op);
    if (!module)
      return op->emitOpError()
             << "has spmd_parameters_sharding but is not a module";
    // Check that the "main" function exists:
    auto main = module.lookupSymbol<func::FuncOp>("main");
    if (!main)
      return module.emitOpError() << "spmd_parameters_sharding: main not found";
    if (main.getNumArguments() != arrayAttr.size())
      return module.emitOpError()
             << "spmd_parameters_sharding: main has " << main.getNumArguments()
             << " arguments, but spmd_parameters_sharding expects "
             << arrayAttr.size();
  }
  return success();
}

}  // namespace mlir::mhlo
