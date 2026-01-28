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

#ifndef ZKX_MLIR_HLO_STABLEHLO_DIALECT_TYPEINFERENCE_H_
#define ZKX_MLIR_HLO_STABLEHLO_DIALECT_TYPEINFERENCE_H_

#include <stdint.h>

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "zkx/mlir_hlo/stablehlo/dialect/Base.h"

namespace mlir::hlo {

//===----------------------------------------------------------------------===//
// Utilities for shape functions
//===----------------------------------------------------------------------===//

bool verifyCompatibleDims(int64_t dimSize1, int64_t dimSize2);

//===----------------------------------------------------------------------===//
// Shape functions for ops.
//===----------------------------------------------------------------------===//
// These functions have been moved out of StablehloOps.cpp in order to be
// shared with the MHLO dialect.
// Because of that, they cannot use any definitions in the StableHLO dialect
// (definitions in Base are fine, because they are shared with MHLO).
// As a result, these definitions (e.g. StableHLO ops and attributes) are
// decomposed into smaller pieces which are passed as individual parameters.
// These parameters have the same names as in the ODS and come in the same
// order in which they are declared in the ODS.

LogicalResult inferAbsOp(std::optional<Location>, Value operand,
                         SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult
inferBroadcastOp(std::optional<Location> location, Value operand,
                 ArrayRef<int64_t> broadcastSizes,
                 SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult inferCaseOp(std::optional<Location> location, Value index,
                          RegionRange branches,
                          SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult
inferClampOp(std::optional<Location> location, Value min, Value operand,
             Value max,
             SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult
inferCompareOp(MLIRContext *context, std::optional<Location>, Value lhs,
               SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult inferConcatenateOp(std::optional<Location> location,
                                 TypeRange inputTypes, int64_t dimension,
                                 SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult inferConstantOp(std::optional<Location>, ElementsAttr value,
                              SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult
inferConvertOp(std::optional<Location> location, Value operand,
               SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult inferCreateTokenOp(HloDialectInterface *dialect,
                                 std::optional<Location> location,
                                 SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult inferDynamicSliceOp(
    std::optional<Location> location, Type operandType,
    TypeRange startIndicesTypes, ArrayRef<int64_t> sliceSizes,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult inferDynamicUpdateSliceOp(
    std::optional<Location> location, Value operand, Value update,
    ValueRange startIndices,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult inferGetDimensionSizeOp(
    std::optional<Location> location, Type operandType, int64_t dimension,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult
inferGetTupleElementOp(std::optional<Location> location, Value operand,
                       int32_t index,
                       SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult inferIfOp(std::optional<Location> location, Value pred,
                        RegionRange branches,
                        SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult
inferMapOp(std::optional<Location> location, ValueRange inputs,
           ArrayRef<int64_t> dimensions, Region &computation,
           SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

// TODO(chokobole): Do we need this? Dependency: interior_padding
LogicalResult inferPadOp(std::optional<Location> location, Type operandType,
                         Type paddingValueType,
                         ArrayRef<int64_t> edgePaddingLow,
                         ArrayRef<int64_t> edgePaddingHigh,
                         SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult
inferReduceOp(std::optional<Location> location, TypeRange inputTypes,
              ArrayRef<int64_t> dimensions, Region &body,
              SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult inferReduceWindowOp(
    std::optional<Location> location, ValueRange inputs, ValueRange initValues,
    ArrayRef<int64_t> windowDimensions,
    std::optional<ArrayRef<int64_t>> windowStrides,
    std::optional<ArrayRef<int64_t>> baseDilations,
    std::optional<ArrayRef<int64_t>> windowDilations,
    std::optional<DenseIntElementsAttr> padding, Region &body,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult inferReverseOp(std::optional<Location> location, Type operandType,
                             SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult inferScatterOp(std::optional<Location> location,
                             ValueRange inputs, Region &update_computation,
                             SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult
inferSelectOp(std::optional<Location> location, Value pred, Value onTrue,
              Value onFalse,
              SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult inferSetDimensionSizeOp(
    HloDialectInterface *dialect, std::optional<Location> location,
    Type operandType, Value size, int64_t dimension,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult inferSliceOp(std::optional<Location> location, Type operandType,
                           ArrayRef<int64_t> startIndices,
                           ArrayRef<int64_t> limitIndices,
                           ArrayRef<int64_t> strides,
                           SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult
inferSortOp(std::optional<Location> location, ValueRange inputs,
            SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

LogicalResult inferTransposeOp(std::optional<Location> loc, Value operand,
                               ArrayRef<int64_t> permutation,
                               SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult inferTupleOp(MLIRContext *context,
                           std::optional<Location> location, ValueRange val,
                           SmallVectorImpl<Type> &inferredReturnTypes);

LogicalResult inferWhileOp(std::optional<Location> location, ValueRange operand,
                           SmallVectorImpl<Type> &inferredReturnTypes);

//===----------------------------------------------------------------------===//
// Verifiers for ops.
//===----------------------------------------------------------------------===//

LogicalResult verifyAddOp(std::optional<Location> location, Operation *op,
                          Type lhsType, Type rhsType, Type resultType);

LogicalResult verifyBitcastConvertOp(std::optional<Location> location,
                                     Value operand, Value result);

LogicalResult verifyBroadcastInDimOp(std::optional<Location> location,
                                     Value operand,
                                     ArrayRef<int64_t> broadcastDimensions,
                                     Value result);

LogicalResult verifyDynamicBroadcastInDimOp(
    std::optional<Location> location, Value operand, Value outputDimensions,
    ArrayRef<int64_t> broadcastDimensions,
    std::optional<ArrayRef<int64_t>> knownExpandingDimensions,
    std::optional<ArrayRef<int64_t>> knownNonexpandingDimensions, Value result);

LogicalResult verifyDynamicIotaOp(std::optional<Location> location,
                                  Value outputShape, int64_t iotaDimension,
                                  Value result);

// TODO(chokobole): Do we need this? Dependency: interior_padding
LogicalResult verifyDynamicPadOp(std::optional<Location> location,
                                 Value operand, Value paddingValue,
                                 Value edgePaddingLow, Value edgePaddingHigh,
                                 Value result);

LogicalResult verifyDynamicReshapeOp(std::optional<Location> location,
                                     Value operand, Value outputShape,
                                     Value result);

LogicalResult verifyIotaOp(std::optional<Location> location,
                           int64_t iotaDimension, Value result);

LogicalResult verifyRealDynamicSliceOp(std::optional<Location> location,
                                       Value operand, Value startIndices,
                                       Value limitIndices, Value strides);

LogicalResult verifyReduceOp(std::optional<Location> location,
                             ValueRange inputs, ValueRange initValues,
                             ArrayRef<int64_t> dimensions, Region &body);

LogicalResult verifyReduceOpInputsAndInferShape(
    std::optional<Location> location, SmallVector<ShapedType> inputTypes,
    ArrayRef<int64_t> dimensions, SmallVector<int64_t> &newDimensions,
    Attribute &encoding);

LogicalResult
verifyReduceWindowOp(std::optional<Location> location, ValueRange inputs,
                     ValueRange initValues, ArrayRef<int64_t> windowDimensions,
                     std::optional<ArrayRef<int64_t>> windowStrides,
                     std::optional<ArrayRef<int64_t>> baseDilations,
                     std::optional<ArrayRef<int64_t>> windowDilations,
                     std::optional<DenseIntElementsAttr> padding, Region &body);

LogicalResult verifyReshapeOp(std::optional<Location> location, Value operand,
                              Value result);

LogicalResult verifyReverseOp(std::optional<Location> location, Value operand,
                              llvm::ArrayRef<int64_t> dimensions);

LogicalResult verifyScatterOp(
    std::optional<Location> location, ValueRange inputs, Value scatterIndices,
    ValueRange updates, ArrayRef<int64_t> updateWindowDims,
    ArrayRef<int64_t> insertedWindowDims, ArrayRef<int64_t> inputBatchingDims,
    ArrayRef<int64_t> scatterIndicesBatchingDims,
    ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim,
    Region &updateComputation);

LogicalResult verifySortOp(std::optional<Location> location, ValueRange inputs,
                           int64_t dimension, Region &comparator);

LogicalResult verifyWhileOp(std::optional<Location> location,
                            ValueRange operand, Region &cond, Region &body);

} // namespace mlir::hlo

#endif // ZKX_MLIR_HLO_STABLEHLO_DIALECT_TYPEINFERENCE_H_
