/* Copyright 2017 The OpenXLA Authors.

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

// Shape inference is used by the ZKX service as the user builds up
// computation requests.

#ifndef ZKX_SERVICE_SHAPE_INFERENCE_H_
#define ZKX_SERVICE_SHAPE_INFERENCE_H_

#include <stdint.h>

#include <optional>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/shape.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

// For a given operation and input shapes, infers what the resulting shape is
// for the operation. With this functionality, the user does not need to specify
// the expected result type for computations that are built up via the API --
// the shape that results from an operation is inferred. Some methods have
// overloads for inferring shape at the HLO level.
//
// TODO(b/73352135): Shape inference does not issue very good error messages, in
// part because HloInstruction::ToString() is not available since shape
// inference runs before the HloInstruction object is created. We need a
// solution for this.
class ShapeInference {
 public:
  // Infers the shape produced by applying the given unary operation to the
  // given input shape.
  static absl::StatusOr<Shape> InferUnaryOpShape(HloOpcode opcode,
                                                 const Shape& shape);
  static absl::StatusOr<Shape> InferUnaryOpShape(HloOpcode opcode,
                                                 const HloInstruction* operand);

  // Infers the shape produced by applying the given binary operation to the
  // given input shapes.
  static absl::StatusOr<Shape> InferBinaryOpShape(
      HloOpcode opcode, const Shape& lhs, const Shape& rhs,
      absl::Span<const int64_t> broadcast_dimensions);
  static absl::StatusOr<Shape> InferBinaryOpShape(HloOpcode opcode,
                                                  const HloInstruction* lhs,
                                                  const HloInstruction* rhs);

  // Infers the shape produced by applying the given ternary operation to the
  // given input shapes.
  static absl::StatusOr<Shape> InferTernaryOpShape(HloOpcode opcode,
                                                   const Shape& lhs,
                                                   const Shape& rhs,
                                                   const Shape& ehs);
  static absl::StatusOr<Shape> InferTernaryOpShape(HloOpcode opcode,
                                                   const HloInstruction* lhs,
                                                   const HloInstruction* rhs,
                                                   const HloInstruction* ehs);

  // Infers the shape produced by applying the given variadic operation to the
  // given input operand shapes.
  static absl::StatusOr<Shape> InferVariadicOpShape(
      HloOpcode opcode, absl::Span<const Shape* const> operand_shapes);
  static absl::StatusOr<Shape> InferVariadicOpShape(
      HloOpcode opcode, absl::Span<const HloInstruction* const> operands);

  // Infers the shape produced by the given FFT type on the given operand.
  static absl::StatusOr<Shape> InferFftShape(const Shape& in, FftType fft_type);

  // Infers the shape produced by an all-gather with the given operand shape,
  // concat dimension, and shard count.
  static absl::StatusOr<Shape> InferAllGatherShape(
      absl::Span<const Shape* const> operand_shapes,
      int64_t all_gather_dimension, int64_t shard_count);

  // Infers the shape produced by an all-gather-start with the given operand
  // shape, concat dimension, and shard count.
  static absl::StatusOr<Shape> InferAllGatherStartShape(
      absl::Span<const Shape* const> operand_shapes,
      int64_t all_gather_dimension, int64_t shard_count);

  // Infers the shape produced by an all-gather-done given a certain
  // all-gather-start shape.
  static absl::StatusOr<Shape> InferAllGatherDoneShape(
      const Shape& all_gather_start_shape);

  // Infers the shape produced by a cross replica sum with the given operand
  // shapes.
  static absl::StatusOr<Shape> InferAllReduceShape(
      absl::Span<const Shape* const> operand_shapes);

  // Infers the shape produced by a reduce-scatter with the given operand
  // shape, scatter dimension, and shard count.
  static absl::StatusOr<Shape> InferReduceScatterShape(
      absl::Span<const Shape* const> operand_shapes, int64_t scatter_dimension,
      int64_t shard_count);

  // Infers the shape produced by a cross replica sum start.
  static absl::StatusOr<Shape> InferAllReduceStartShape(
      absl::Span<const Shape* const> operand_shapes);

  // Infers the shape produced by a cross replica sum done.
  static absl::StatusOr<Shape> InferAllReduceDoneShape(
      const Shape& operand_shape);

  // Infers final shape of an Alltoall operation that is created by the xla
  // builder.
  static absl::StatusOr<Shape> InferAllToAllShape(const Shape& shape,
                                                  int64_t split_dimension,
                                                  int64_t concat_dimension,
                                                  int64_t split_count);

  // Infers the shape of an HLO all-to-all instruction.
  static absl::StatusOr<Shape> InferAllToAllTupleShape(
      absl::Span<const Shape* const> operand_shapes);

  // Infers the shape of an HLO ragged-all-to-all instruction.
  static absl::StatusOr<Shape> InferRaggedAllToAllShape(
      absl::Span<const Shape* const> operand_shapes);

  // Infers the shape of a collective broadcast operation.
  static absl::StatusOr<Shape> InferCollectiveBroadcastShape(
      absl::Span<const Shape* const> operand_shapes);

  // Infers the shape of a collective permute operation.
  static absl::StatusOr<Shape> InferCollectivePermuteShape(
      absl::Span<const Shape* const> operand_shapes, bool inplace);

  // Infers the shape of a collective permute start operation.
  static absl::StatusOr<Shape> InferCollectivePermuteStartShape(
      absl::Span<const Shape* const> operand_shapes,
      absl::Span<const Shape> context_shapes, bool inplace);

  // Infers the shape of a collective permute operation.
  static absl::StatusOr<Shape> InferCollectivePermuteDoneShape(
      const Shape& operand_shape);

 private:
  // Helper that infers the shape produced by performing an element-wise binary
  // operation with the given LHS and RHS shapes.
  // Note: By "element-wise" we mean operations that look at a single element in
  // the LHS and a single element in the RHS to produce a single output element,
  // even in the presence of broadcasting of one of the operands over the other.
  static absl::StatusOr<Shape> InferElementwiseBinaryOpShape(
      HloOpcode operation, const Shape& lhs, const Shape& rhs,
      absl::Span<const int64_t> broadcast_dimensions);

  // Helper for inferring the shape of Select ops.
  static absl::StatusOr<Shape> InferSelectShape(const Shape& pred,
                                                const Shape& on_true,
                                                const Shape& on_false);

  // Helper for inferring shapes of binary operations which use degenerate
  // dimension broadcasting (a dimension of size 1 in one operand is broadcast
  // up to match the size of the dimension in the other operand).
  static absl::StatusOr<Shape> InferDegenerateDimensionBroadcastShape(
      const Shape& lhs, const Shape& rhs);

  // Helper for inferring shapes of binary operations using "InDim"
  // broadcasting. This is the broadcasting used in the *InDim binary operations
  // (for example ComputationBuilder::AddInDim). smaller_shape must be a
  // lower-rank shape than larger_shape. Returns the shape that the
  // smaller_shape is broadcast to.
  //
  // Since this method is only used by InferBinaryOpShape transitively, this
  // method also supports inference of unbounded dynamic dimensions.
  static absl::StatusOr<Shape> InferInDimBroadcastShape(
      const Shape& smaller_shape, const Shape& larger_shape,
      absl::Span<const int64_t> broadcast_dimensions);

  ShapeInference(const ShapeInference&) = delete;
  ShapeInference& operator=(const ShapeInference&) = delete;
};

}  // namespace zkx

#endif  // ZKX_SERVICE_SHAPE_INFERENCE_H_
