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

#include "zkx/mlir_hlo/stablehlo/dialect/TypeInference.h"

#include "llvm/Support/MathExtras.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"

#include "zkx/mlir_hlo/stablehlo/dialect/Base.h"

namespace mlir::hlo {

LogicalResult inferConstantOp(std::optional<Location>, ElementsAttr value,
                              SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(value.getType());
  return success();
}

LogicalResult inferConvertOp(
    std::optional<Location> location, Value operand,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto operandType = cast<ShapedType>(operand.getType());
  // convert_c1
  inferredReturnShapes.emplace_back(operandType.getShape());
  return success();
}

LogicalResult inferSliceOp(std::optional<Location> location, Type operandType,
                           ArrayRef<int64_t> startIndices,
                           ArrayRef<int64_t> limitIndices,
                           ArrayRef<int64_t> strides,
                           SmallVectorImpl<Type>& inferredReturnTypes) {
  auto rankedTy = cast<RankedTensorType>(operandType);

  // slice_c2
  int64_t rank = rankedTy.getRank();
  if (static_cast<int64_t>(startIndices.size()) != rank)
    return emitOptionalError(
        location, "the number of elements in start_indices (",
        startIndices.size(), ") does not match the rank of the operand (", rank,
        ")");

  ArrayRef<int64_t> inputBounds = encodingToBounds(rankedTy.getEncoding());
  SmallVector<int64_t> shape(rank, ShapedType::kDynamic);
  SmallVector<int64_t> resultBounds(inputBounds.size(), ShapedType::kDynamic);

  for (int64_t i = 0, e = rank; i != e; i++) {
    // slice_c3
    if (startIndices[i] < 0)
      return emitOptionalError(location, "negative start index ",
                               startIndices[i], " in dimension ", i);

    bool isStaticDim = isStaticDimSize(rankedTy.getDimSize(i));
    bool isStaticBound =
        !inputBounds.empty() && isStaticDimSize(inputBounds[i]);
    if (isStaticDim || isStaticBound) {
      int64_t operandSizeOrBound =
          isStaticDim ? rankedTy.getDimSize(i) : inputBounds[i];
      StringRef sizeOrBound = isStaticDim ? "size" : "bound";
      // slice_c3
      if (limitIndices[i] > operandSizeOrBound)
        return emitOptionalError(location, "limit index ", limitIndices[i],
                                 " is larger than dimension ", sizeOrBound, " ",
                                 operandSizeOrBound, " in dimension ", i);
    }

    // slice_c3
    if (startIndices[i] > limitIndices[i])
      return emitOptionalError(location, "start index ", startIndices[i],
                               " is larger than limit index ", limitIndices[i],
                               " in dimension ", i);
    // slice_c4
    if (strides[i] <= 0)
      return emitOptionalError(location, "stride must be positive but got ",
                               strides[i], " in dimension ", i);

    // slice_c5
    shape[i] = static_cast<int64_t>(
        llvm::divideCeil(limitIndices[i] - startIndices[i], strides[i]));
  }

  // slice_c1
  inferredReturnTypes.push_back(RankedTensorType::get(
      shape, rankedTy.getElementType(),
      boundsToEncoding(rankedTy.getEncoding(), resultBounds)));
  return success();
}

LogicalResult inferTupleOp(MLIRContext* context, std::optional<Location>,
                           ValueRange val,
                           SmallVectorImpl<Type>& inferredReturnTypes) {
  // tuple_c1
  inferredReturnTypes.push_back(TupleType::get(context, val.getTypes()));
  return success();
}

}  // namespace mlir::hlo
