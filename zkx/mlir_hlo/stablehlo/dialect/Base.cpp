/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#include "zkx/mlir_hlo/stablehlo/dialect/Base.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

// Include order matters
#include "zkx/mlir_hlo/stablehlo/dialect/BaseAttrInterfaces.cpp.inc"

#define DEBUG_TYPE "stablehlo-base"

namespace mlir::hlo {

LogicalResult verifyCompatibleShapeWithBounds(Type type1, Type type2) {
  if (failed(verifyCompatibleShape(type1, type2))) return failure();

  // Verify shapes against bounds
  auto isCompatible = [](ArrayRef<int64_t> shape,
                         BoundedAttrInterface boundedAttr) {
    if (shape.empty() || !boundedAttr) return true;
    auto bounds = boundedAttr.getBounds();
    for (auto [dim_size, bound] : llvm::zip(shape, bounds))  // NOLINT
      if (isStaticDimSize(bound) && bound < dim_size) return false;
    return true;
  };

  auto rankedType1 = dyn_cast<RankedTensorType>(type1);
  auto rankedType2 = dyn_cast<RankedTensorType>(type2);
  if (rankedType1 && rankedType2) {
    auto boundedAttr1 =
        dyn_cast_or_null<BoundedAttrInterface>(rankedType1.getEncoding());
    auto boundedAttr2 =
        dyn_cast_or_null<BoundedAttrInterface>(rankedType2.getEncoding());
    return LogicalResult::success(
        isCompatible(rankedType1.getShape(), boundedAttr2) &&
        isCompatible(rankedType2.getShape(), boundedAttr1));
  }
  return success();
}

bool isCompatibleElementTypeForHloTypeInference(Type tp1, Type tp2) {
  // Get element type if shaped
  tp1 = getElementTypeOrSelf(tp1);
  tp2 = getElementTypeOrSelf(tp2);

  // Sparsity: In the most general case, we allow any combination of
  // sparsity/denseness across any combination of operands/results, as well as
  // differences in sparsity encodings for operands and results.
  // Individual ops may introduce additional constraints.
  // No additional code is needed to check this because of how sparsity is
  // currently implemented.

  // Default case: Unless dynamism and/or sparsity are involved,
  // the types are required to be exactly equal.
  return tp1 == tp2;
}

bool isCompatibleForHloTypeInference(Type tp1, Type tp2) {
  // Dynamism: We don't require shapes to be the same, we only require them
  // to be compatible, which means that:
  //   1) At least one of the shapes is unranked.
  //   2) Or both shapes have the same rank and their dimensions are compatible,
  //     i.e. for each pair of corresponding dimensions:
  //       2.1) At least one of the dimensions is dynamic,
  //       2.2) Or both dimensions are equal.
  // These relaxed rules simplify the implementation of type inference, allowing
  // ops with partially inferred types to pass verification.
  auto stp1 = dyn_cast<ShapedType>(tp1);
  auto stp2 = dyn_cast<ShapedType>(tp2);
  if (stp1 && stp2)
    return succeeded(verifyCompatibleShapeWithBounds(stp1, stp2)) &&
           isCompatibleElementTypeForHloTypeInference(stp1.getElementType(),
                                                      stp2.getElementType());

  return isCompatibleElementTypeForHloTypeInference(tp1, tp2);
}

bool isCompatibleForHloTypeInference(TypeRange tp1, TypeRange tp2) {
  if (tp1.size() != tp2.size()) return false;
  for (auto [lt, rt] : llvm::zip(tp1, tp2))
    if (!isCompatibleForHloTypeInference(lt, rt)) return false;
  return true;
}

bool isCompatibleForHloTypeInference(ArrayRef<int64_t> shape1, Type tp2) {
  // NOLINTNEXTLINE(readability/braces)
  if (llvm::any_of(shape1, [&](int64_t x) { return x < 0; })) return false;
  auto stp2 = dyn_cast<ShapedType>(tp2);
  if (!stp2) return false;
  return isCompatibleForHloTypeInference(
      RankedTensorType::get(shape1, stp2.getElementType()), tp2);
}

bool isCompatibleForHloTypeInference(Value shape1, Type tp2) {
  SmallVector<int64_t> shapeVec1;
  if (!succeeded(matchInts(shape1, shapeVec1))) return true;
  return isCompatibleForHloTypeInference(shapeVec1, tp2);
}

LogicalResult matchInt(Value value, int64_t& result) {
  APInt constValue;
  if (!matchPattern(value, m_ConstantInt(&constValue))) return failure();
  result = constValue.getSExtValue();
  return success();
}

LogicalResult matchInts(Value value, SmallVector<int64_t>& result) {
  DenseIntElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) return failure();
  for (auto element : attr.getValues<APInt>()) {
    result.push_back(element.getSExtValue());
  }
  return success();
}

LogicalResult matchInts(Value value, SmallVector<APSInt>& result) {
  DenseIntElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) return failure();

  // Signless types are treated as signed, per StableHLO convention.
  // Unless the type is i1 (which models boolean type from the StableHLO spec),
  // in which case it's considered to be unsigned.
  Type elementType = attr.getType().getElementType();
  bool isUnsigned = elementType.isUnsignedInteger() || elementType.isInteger(1);

  for (auto element : attr.getValues<APInt>()) {
    result.push_back(APSInt(element, /*isUnsigned=*/isUnsigned));
  }
  return success();
}

LogicalResult matchInts(Value value) {
  DenseIntElementsAttr attr;
  return success(/*isSuccess=*/matchPattern(value, m_Constant(&attr)));
}

LogicalResult deriveShapeFromOperand(
    OpBuilder* builder, Operation* op, Value operand,
    SmallVectorImpl<Value>* reifiedReturnShapes) {
  auto shapedTy = dyn_cast<ShapedType>(operand.getType());
  if (!shapedTy) {
    op->emitOpError() << "operand is not a shaped type";
    return failure();
  }
  reifiedReturnShapes->assign(
      {builder->create<shape::ShapeOfOp>(op->getLoc(), operand)});
  return success();
}

//===----------------------------------------------------------------------===//
// Utils for shape functions with bounded dynamism.
//===----------------------------------------------------------------------===//

LogicalResult verifyBounds(ArrayRef<int64_t> bounds, RankedTensorType type,
                           function_ref<InFlightDiagnostic()> emitError) {
  int64_t boundsLen = bounds.size();
  int64_t rank = type.getRank();
  if (boundsLen != rank)
    return emitError() << "Bounds length is " << boundsLen
                       << ", expected to be equal to rank(" << rank
                       << ") of the tensor";

  for (int64_t dim = 0; dim < rank; ++dim) {
    int64_t bound = bounds[dim];
    int64_t dimSize = type.getDimSize(dim);
    if (bound != ShapedType::kDynamic && dimSize != ShapedType::kDynamic)
      return emitError() << "Static dimension " << dim
                         << " cannot have a bound, use ShapedType::kDynamic to "
                            "indicate a missing bound";
  }

  return success();
}

ArrayRef<int64_t> encodingToBounds(Attribute encoding) {
  if (auto boundedAttr = dyn_cast_or_null<BoundedAttrInterface>(encoding))
    return boundedAttr.getBounds();
  return {};
}

Attribute boundsToEncoding(Attribute prototype, ArrayRef<int64_t> bounds) {
  if (bounds.empty()) return prototype;
  if (llvm::all_of(bounds, [&](auto b) { return isDynamicDimSize(b); }))
    return {};
  if (!prototype)
    llvm::report_fatal_error(
        "Expect an prototype attribute to obtain the underlying dialect but "
        "got none");
  auto dialect = cast<HloDialectInterface>(&prototype.getDialect());
  return dialect->createTypeExtensions(bounds);
}

FailureOr<std::pair<int64_t, int64_t>> inferMostSpecificDimAndBound(
    std::optional<Location> location, int64_t dim, int64_t leftSize,
    int64_t rightSize, int64_t leftBound, int64_t rightBound) {
  bool isLeftStaticDim = isStaticDimSize(leftSize);
  bool isRightStaticDim = isStaticDimSize(rightSize);
  bool isLeftStaticBound = isStaticDimSize(leftBound);
  bool isRightStaticBound = isStaticDimSize(rightBound);
  int64_t inferredSize = ShapedType::kDynamic;
  int64_t inferredBound = ShapedType::kDynamic;

  if (isLeftStaticDim || isRightStaticDim) {
    if (isLeftStaticDim && isRightStaticDim && leftSize != rightSize)
      return emitOptionalError(location, "Mismatched dimension sizes ",
                               leftSize, " and ", rightSize, " in dimension ",
                               dim);
    inferredSize = isLeftStaticDim ? leftSize : rightSize;
    if (isLeftStaticBound || isRightStaticBound) {
      int64_t check_bound = isLeftStaticBound ? leftBound : rightBound;
      if (inferredSize > check_bound)
        return emitOptionalError(location, "Mismatched dimension size ",
                                 inferredSize, " and bound ", check_bound,
                                 " in dimension ", dim);
    }
  } else {
    if (isLeftStaticBound && isRightStaticBound)
      inferredBound = std::min(leftBound, rightBound);
    else
      inferredBound = isLeftStaticBound ? leftBound : rightBound;
  }
  return std::make_pair(inferredSize, inferredBound);
}

FailureOr<std::pair<int64_t, int64_t>> inferLeastSpecificDimAndBound(
    std::optional<Location> location, int64_t dim, int64_t leftSize,
    int64_t rightSize, int64_t leftBound, int64_t rightBound) {
  bool isLeftStaticDim = isStaticDimSize(leftSize);
  bool isRightStaticDim = isStaticDimSize(rightSize);
  bool isLeftStaticBound = isStaticDimSize(leftBound);
  bool isRightStaticBound = isStaticDimSize(rightBound);
  int64_t inferredSize = ShapedType::kDynamic;
  int64_t inferredBound = ShapedType::kDynamic;

  if (isLeftStaticDim || isRightStaticDim) {
    if (isLeftStaticDim && isRightStaticDim) {
      if (leftSize != rightSize)
        return emitOptionalError(location, "Mismatched dimension sizes ",
                                 leftSize, " and ", rightSize, " in dimension ",
                                 dim);
      inferredSize = leftSize;
    } else if (isLeftStaticBound || isRightStaticBound) {
      inferredBound = isLeftStaticDim ? std::max(leftSize, rightBound)
                                      : std::max(rightSize, leftBound);
    }
  } else if (isLeftStaticBound && isRightStaticBound) {
    inferredBound = std::max(leftBound, rightBound);
  }
  return std::make_pair(inferredSize, inferredBound);
}

namespace {

FailureOr<ShapedType> inferTypeWithCustomFn(
    std::optional<Location> location, SmallVector<RankedTensorType> rankedTypes,
    std::function<FailureOr<std::pair<int64_t, int64_t>>(
        std::optional<Location>, int64_t, int64_t, int64_t, int64_t, int64_t)>
        inferDimAndBoundFn) {
  auto rank = rankedTypes[0].getRank();
  for (RankedTensorType type : rankedTypes) {
    if (type.getRank() != rank) {
      return emitOptionalError(location, "Mismatched ranks of types",
                               rankedTypes[0].getRank(), " vs ",
                               type.getRank());
    }
  }
  SmallVector<int64_t> inferredSizes = to_vector(rankedTypes[0].getShape());
  SmallVector<int64_t> inferredBounds(rank, ShapedType::kDynamic);
  ArrayRef<int64_t> bounds = encodingToBounds(rankedTypes[0].getEncoding());
  if (!bounds.empty()) inferredBounds = to_vector(bounds);
  bool anyInputHaveBounds = !bounds.empty();

  for (size_t i = 1; i < rankedTypes.size(); ++i) {
    bounds = encodingToBounds(rankedTypes[i].getEncoding());
    for (int dim = 0; dim < rank; ++dim) {
      FailureOr<std::pair<int64_t, int64_t>> inferredDimAndBoundOrErr =
          inferDimAndBoundFn(location, dim,
                             /*leftSize=*/inferredSizes[dim],
                             /*rightSize=*/rankedTypes[i].getShape()[dim],
                             /*leftBound=*/inferredBounds[dim],
                             /*rightBound=*/bounds.empty()
                                 ? ShapedType::kDynamic
                                 : bounds[dim]);
      if (failed(inferredDimAndBoundOrErr)) return failure();
      inferredSizes[dim] = (*inferredDimAndBoundOrErr).first;
      inferredBounds[dim] = (*inferredDimAndBoundOrErr).second;
    }
  }

  return {RankedTensorType::get(
      inferredSizes, rankedTypes[0].getElementType(),
      boundsToEncoding(
          rankedTypes[0].getEncoding(),
          // Empty array as argument is an indicator to boundsToEncoding() that
          // there are no bounds at all in inputs, thus sparsity attributes will
          // be included in the return type
          anyInputHaveBounds ? inferredBounds : ArrayRef<int64_t>({})))};
}

FailureOr<Type> inferLeastSpecificShapedType(std::optional<Location> location,
                                             TypeRange inputTypes) {
  SmallVector<RankedTensorType> rankedTypes;
  for (Type inputType : inputTypes)
    if (auto rankedType = dyn_cast<RankedTensorType>(inputType))
      rankedTypes.push_back(rankedType);
    else
      return inputType;
  return inferTypeWithCustomFn(location, rankedTypes,
                               inferLeastSpecificDimAndBound);
}

FailureOr<Type> inferMostSpecificShapedType(std::optional<Location> location,
                                            TypeRange inputTypes) {
  SmallVector<RankedTensorType> rankedTypes;
  for (Type inputType : inputTypes)
    if (auto rankedType = dyn_cast<RankedTensorType>(inputType))
      rankedTypes.push_back(rankedType);
  if (rankedTypes.empty()) return inputTypes[0];
  return inferTypeWithCustomFn(location, rankedTypes,
                               inferMostSpecificDimAndBound);
}

// Applies `fn` to `inputTypes`, using `location` for errors.
// If `inputTypes` are tuples, then applies `fn` to them elementwise and
// wraps the results into a tuple, for example:
//   mapOverTupleElements({tuple<T11, T12>, tuple<T21, T22>}, fn) =
//     tuple<fn(T11, T21), fn(T12, T22)>
// Only supports `inputTypes` where either all types are tuples or no types
// are tuples.
FailureOr<Type> mapOverTupleElements(
    std::optional<Location> location, TypeRange inputTypes,
    function_ref<FailureOr<Type>(std::optional<Location>, TypeRange types)>
        fn) {
  SmallVector<TupleType> tupleTypes;
  for (Type inputType : inputTypes) {
    if (auto tupleType = dyn_cast<TupleType>(inputType))
      tupleTypes.push_back(tupleType);
  }
  if (!tupleTypes.empty()) {
    if (tupleTypes.size() != inputTypes.size())
      return emitOptionalError(location,
                               "Mismatched type kinds: either all types ",
                               "must be tuples, or no types must be tuples");
    SmallVector<Type> results(tupleTypes[0].size());
    for (TupleType tupleType : tupleTypes) {
      if (tupleType.size() != results.size())
        return emitOptionalError(location,
                                 "Mismatched tuple sizes: all tuple sizes ",
                                 "must be the same");
    }
    for (size_t i = 0; i < results.size(); ++i) {
      SmallVector<Type> ithElements;
      for (TupleType tupleType : tupleTypes)
        ithElements.push_back(tupleType.getType(i));
      FailureOr<Type> result = fn(location, ithElements);
      if (failed(result)) return failure();
      results[i] = *result;
    }
    return TupleType::get(tupleTypes[0].getContext(), results);
  }
  return fn(location, inputTypes);
}

}  // namespace

FailureOr<Type> inferLeastSpecificType(std::optional<Location> location,
                                       TypeRange inputTypes) {
  return mapOverTupleElements(location, inputTypes,
                              inferLeastSpecificShapedType);
}

FailureOr<Type> inferMostSpecificType(std::optional<Location> location,
                                      TypeRange inputTypes) {
  return mapOverTupleElements(location, inputTypes,
                              inferMostSpecificShapedType);
}

bool hasSingleBoundedDimension(Type type) {
  RankedTensorType rankedType = dyn_cast<RankedTensorType>(type);
  auto boundedAttr =
      dyn_cast_or_null<BoundedAttrInterface>(rankedType.getEncoding());
  if (!boundedAttr) return false;

  // Count if bounded attr size is not kDynamic
  int64_t numBoundedDims = llvm::count_if(
      boundedAttr.getBounds(),
      [](int64_t bound) { return !ShapedType::isDynamic(bound); });
  // Also check that there are only bounded dims and no unbounded dims.
  int64_t numDynamicDims = llvm::count_if(
      rankedType.getShape(),
      [](int64_t bound) { return ShapedType::isDynamic(bound); });
  return numBoundedDims == 1 && numDynamicDims == 1;
}

}  // namespace mlir::hlo
