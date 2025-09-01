/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#ifndef ZKX_MLIR_HLO_STABLEHLO_DIALECT_BASE_H_
#define ZKX_MLIR_HLO_STABLEHLO_DIALECT_BASE_H_

#include <stdint.h>

#include <optional>

#include "llvm/ADT/APSInt.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

// Include order matters
#include "zkx/mlir_hlo/stablehlo/dialect/BaseAttrInterfaces.h.inc"

namespace mlir::hlo {
namespace {

// TODO(zhouxin) change to a better name as it's used by both of size and bound
// Check if the dimension size is dynamic.
inline bool isDynamicDimSize(int64_t val) { return ShapedType::isDynamic(val); }

inline bool isStaticDimSize(int64_t val) { return !isDynamicDimSize(val); }

}  // namespace

// Checks whether every position in the given array contains the given value.
bool isSplatArray(ArrayRef<int64_t> arr, int64_t val);

//  Verifies that the two types have compatible shape with bounds but allows
//  different element types.
LogicalResult verifyCompatibleShapeWithBounds(Type type1, Type type2);

// Returns true if the given element types are compatible for the purposes of
// HLO type inference, accounting for special properties of quantization and
// sparsity.
bool isCompatibleElementTypeForHloTypeInference(Type tp1, Type tp2);

// Returns true if the given types are compatible for the purposes of HLO type
// inference, accounting for special properties of dynamism, quantization and
// sparsity.
bool isCompatibleForHloTypeInference(Type tp1, Type tp2);

// Returns true if the given type ranges are compatible for the purposes of HLO
// type inference, accounting for special properties of dynamism, quantization
// and sparsity.
bool isCompatibleForHloTypeInference(TypeRange tp1, TypeRange tp2);

// Returns true if the given shape, expressed as a runtime value, is compatible
// with the given type for the purposes of HLO type inference.
// If we know that this runtime value is a constant, then we perform the check.
// If we don't, then we return true - because shape mismatches at runtime are
// undefined behavior.
bool isCompatibleForHloTypeInference(Value shape1, Type tp2);

// Returns true if the given shape, expressed as a slice of integers, is
// compatible with the given type for the purposes of HLO type inference.
bool isCompatibleForHloTypeInference(ArrayRef<int64_t> shape1, Type tp2);

// Inference rules to merge dimensions with bounds (lhs/rhs are commutative):
//       Dim of lhs     Dim of rhs      Infer
//  c0:  X              X               X
//  c1:  X              ?               X
//  c2:  X              ?, B(>=X)       X
//  c3:  X              ?, B(<X)        Will error out by compatible checks
//  c4:  ?              ?               ?
//  c5:  ?              ?, B            ?, B
//  c6:  ?, B           ?, C            ?, min(B, C)
FailureOr<std::pair<int64_t, int64_t>> inferMostSpecificDimAndBound(
    std::optional<Location> location, int64_t dim, int64_t leftSize,
    int64_t rightSize, int64_t leftBound, int64_t rightBound);

// Inference rules for conditional branches (lhs/rhs are commutative):
//       Dim of lhs     Dim of rhs      Infer
//  c0:  X              X               X
//  c1:  X              ?               ?
//  c2:  X              ?, B            ?, max(X, B)
//  c3:  ?              ?               ?
//  c4:  ?              ?, B            ?
//  c5:  ?, B           ?, C            ?, max(B, C)
FailureOr<std::pair<int64_t, int64_t>> inferLeastSpecificDimAndBound(
    std::optional<Location> location, int64_t dim, int64_t leftSize,
    int64_t rightSize, int64_t leftBound, int64_t rightBound);

// Infer single least specific return type from inputTypes with support for
// bounds. (Size, bound) of each dimension of the return type will be merged
// from corresponding dimensions of every inputType by extracting the least
// specific one. Return unranked tensor if any input is unranked.
FailureOr<Type> inferLeastSpecificType(std::optional<Location> location,
                                       TypeRange inputTypes);

// Infer single most specific return type from inputTypes with support for
// bounds. (Size, bound) of each dimension of the return type will be merged
// from corresponding dimensions of every inputType by extracting the most
// specific one. Return unranked tensor if all inputs are unranked.
FailureOr<Type> inferMostSpecificType(std::optional<Location> location,
                                      TypeRange inputTypes);

LogicalResult inferMostSpecificTypeComponents(
    std::optional<Location> location, TypeRange inputTypes,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes);

// Matches a constant with integer value into int64_t.
LogicalResult matchInt(Value value, int64_t &result);

// Matches a constant tensor with integer values into a 1-dimensional vector.
// Doesn't preserve the bitness or the signedness of the underlying values,
// extracting them into int64_t.
LogicalResult matchInts(Value value, SmallVector<int64_t> &result);

// Matches a constant tensor with integer values into a 1-dimensional vector.
// Preserves the bitness and the signedness of the underlying values.
LogicalResult matchInts(Value value, SmallVector<APSInt> &result);

// Matches a constant tensor with integer values.
// Unlike the functions above, it doesn't return these values - it just checks
// that the given argument is indeed a constant tensor with integer values.
LogicalResult matchInts(Value value);

// Verify bounds expressed by HLO_BoundedAttrInterface against the provided
// type. See documentation for HLO_BoundedAttrInterface for the list of checks.
LogicalResult verifyBounds(ArrayRef<int64_t> bounds, RankedTensorType type,
                           function_ref<InFlightDiagnostic()> emitError);

// If an encoding attribute conforms to HLO_BoundedAttrInterface, return the
// bounds that it carries. Otherwise, return an empty ArrayRef.
ArrayRef<int64_t> encodingToBounds(Attribute encoding);

// Create an HLO_BoundedAttrInterface encoding attribute that carries the given
// bounds. Requires a prototype - an existing encoding attribute - to obtain
// the underlying dialect that knows how to create these attributes.
Attribute boundsToEncoding(Attribute prototype, ArrayRef<int64_t> bounds);

// This interface is implemented by both StableHLO and MHLO dialects
// and is used as the foundation for sharing verification, type inference and
// prettyprinting logic between them.
class HloDialectInterface : public DialectInterface::Base<HloDialectInterface> {
 public:
  HloDialectInterface(Dialect *dialect) : Base(dialect) {}

  // Creates a TokenType type, specific to this dialect.
  // See docs for the particular type in the corresponding dialect.
  virtual Type createTokenType() const = 0;

  // Check whether the type is of TokenType in the corresponding dialect.
  virtual bool isTokenType(Type type) const = 0;

  // Creates a TypeExtensions attribute, specific to this dialect.
  // See docs for the particular attribute in the corresponding dialect.
  virtual Attribute createTypeExtensions(ArrayRef<int64_t> bounds) const = 0;
};

// Returns true if the given type has a single bounded dimension.
bool hasSingleBoundedDimension(Type type);

}  // namespace mlir::hlo

#endif  // ZKX_MLIR_HLO_STABLEHLO_DIALECT_BASE_H_
