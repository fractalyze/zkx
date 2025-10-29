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

#include "zkx/service/shape_inference.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/hlo/ir/hlo_instructions.h"

namespace zkx {
namespace {

// Checks whether the given dimension size `size` is unbounded dynamic size.
bool IsUnboundedDynamicSize(int64_t size) {
  return size == Shape::kUnboundedSize;
}

// Returns success if the given two dimension sizes 'size_a' and 'size_b' are
// compatible: at least one is dynamic or both are equal.
bool CompatibleDimensionSizes(int64_t size_a, int64_t size_b) {
  return IsUnboundedDynamicSize(size_a) || IsUnboundedDynamicSize(size_b) ||
         size_a == size_b;
}

absl::Status ExpectArray(const Shape& shape, std::string_view op_type) {
  if (!shape.IsArray()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected array argument for %s, but got %s.", op_type,
                        ShapeUtil::HumanString(shape)));
  }
  return absl::OkStatus();
}

// Encapsulates inferred dimension size and bound size.
struct DimAndBound {
  int64_t dimension, bound;
};

// Inference rules to merge dimensions with bounds (lhs/rhs are commutative):
//       Dim of lhs     Dim of rhs      Infer
//  c0:  X              X               X
//  c1:  X              ?               X
//  c2:  X              <=X             <=X
//  c3:  ?              ?               ?
//  c4:  ?              <=B             <=B
//  c5:  <=B            <=C             Error, mismatched bound sizes
//  c6:  X              Y               Error, mismatched dimension sizes
// Note:
// A HLO static dimension size `X` is expressed as size=X, and bound=?
// A bounded dynamic dimension size `<=X` is be expressed as size=X, and bound=?
// A unbounded dynamic dimension size, `?`, is expressed as size=?, and bound=?
absl::StatusOr<DimAndBound> InferMostSpecificDimAndBound(int64_t dim,
                                                         int64_t left_size,
                                                         int64_t right_size,
                                                         int64_t left_bound,
                                                         int64_t right_bound) {
  bool is_left_static_dim = !IsUnboundedDynamicSize(left_size);
  bool is_right_static_dim = !IsUnboundedDynamicSize(right_size);
  bool is_left_static_bound = !IsUnboundedDynamicSize(left_bound);
  bool is_right_static_bound = !IsUnboundedDynamicSize(right_bound);
  int64_t inferred_size = Shape::kUnboundedSize;
  int64_t inferred_bound = Shape::kUnboundedSize;

  if (is_left_static_bound || is_right_static_bound) {
    if (is_left_static_bound && is_right_static_bound &&
        left_bound != right_bound) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Mismatched bound sizes %d and %d in dimension %d",
                          left_bound, right_bound, dim));
    }
    inferred_bound = is_left_static_bound ? left_bound : right_bound;
  }
  if (is_left_static_dim || is_right_static_dim) {
    if (is_left_static_dim && is_right_static_dim && left_size != right_size) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Mismatched dimension sizes %d and %d in dimension %d", left_size,
          right_size, dim));
    }
    inferred_size = is_left_static_dim ? left_size : right_size;
    if (!IsUnboundedDynamicSize(inferred_bound) &&
        inferred_size != inferred_bound) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Mismatched dimension size %d and bound %d in dimension %d",
          inferred_size, inferred_bound, dim));
    }
  }
  DimAndBound dim_and_bound = {inferred_size, inferred_bound};
  return dim_and_bound;
}

}  // namespace

// static
absl::StatusOr<Shape> ShapeInference::InferUnaryOpShape(
    HloOpcode opcode, const HloInstruction* operand) {
  return InferUnaryOpShape(opcode, operand->shape());
}

// static
absl::StatusOr<Shape> ShapeInference::InferUnaryOpShape(HloOpcode opcode,
                                                        const Shape& shape) {
  // There is no copy operation at the proto level, so handle copy explicitly.
  // A domain shape is the same as the input one.
  if (opcode == HloOpcode::kCopy || opcode == HloOpcode::kDomain) {
    return shape;
  }

  TF_RETURN_IF_ERROR(ExpectArray(shape, "operand of unary operation"));

  DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape));
  switch (opcode) {
      // TODO(chokobole): Uncomment this. Dependency: case HloOpcode::kAbs
      // case HloOpcode::kAbs:

      // TODO(chokobole): Uncomment this. Dependency: case HloOpcode::kClz
    // case HloOpcode::kClz:
    case HloOpcode::kInverse:
      if (!ShapeUtil::ElementIsField(shape)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Expected element type in shape to be field for "
                            "inverse operation; got %s.",
                            PrimitiveType_Name(shape.element_type())));
      }
      return shape;
    case HloOpcode::kNegate:
      if (!ShapeUtil::ElementIsIntegral(shape)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Expected element type in shape to be integral for "
                            "negate operation; got %s.",
                            PrimitiveType_Name(shape.element_type())));
      }
      return shape;
      // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: case HloOpcode::kPopulationCount
      // clang-format on
      // case HloOpcode::kPopulationCount:
      // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: case HloOpcode::kSign
      // clang-format on
      // case HloOpcode::kSign:

      // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: case HloOpcode::kNot
      // clang-format on
      // case HloOpcode::kNot:

    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unknown operation for unary shape inference: \"%s\".",
          HloOpcodeString(opcode)));
  }
}

// Current DotDimensionNumbers Requirements:
//
// Contracting Dimensions:
// *) Same number of contracting dimensions on both lhs and rhs.
// *) Contracting dimension size must be the same on both lhs and rhs.
//
// Batch Dimensions:
// *) Same number of batch dimensions on both lhs and rhs.
// *) Same batch dimension sizes on both lhs and rhs.
//

namespace {

absl::Status ValidateDotDimensionNumbers(
    const Shape& lhs, const Shape& rhs,
    const DotDimensionNumbers& dimension_numbers) {
  // Check that dimension numbers are in range.
  auto dims_in_range = [](const int64_t rank,
                          absl::Span<const int64_t> contracting_dims,
                          absl::Span<const int64_t> batch_dims) -> bool {
    auto in_range = [&rank](int64_t i) -> bool { return 0 <= i && i < rank; };
    return absl::c_all_of(contracting_dims, in_range) &&
           absl::c_all_of(batch_dims, in_range);
  };

  absl::Span<const int64_t> lhs_contracting_dimensions =
      dimension_numbers.lhs_contracting_dimensions();
  absl::Span<const int64_t> rhs_contracting_dimensions =
      dimension_numbers.rhs_contracting_dimensions();
  absl::Span<const int64_t> lhs_batch_dimensions =
      dimension_numbers.lhs_batch_dimensions();
  absl::Span<const int64_t> rhs_batch_dimensions =
      dimension_numbers.rhs_batch_dimensions();

  if (!dims_in_range(lhs.rank(), lhs_contracting_dimensions,
                     lhs_batch_dimensions) ||
      !dims_in_range(rhs.rank(), rhs_contracting_dimensions,
                     rhs_batch_dimensions)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "A dimension number is out of range in Dot: %s. %s %s",
        dimension_numbers.DebugString(), lhs.ToString(), rhs.ToString()));
  }

  // Check that dimension numbers are unique.
  auto dims_unique = [](absl::Span<const int64_t> contracting_dims,
                        absl::Span<const int64_t> batch_dims) -> bool {
    absl::flat_hash_set<int64_t> dim_set;
    auto is_unique = [&dim_set](int64_t i) -> bool {
      return dim_set.insert(i).second;
    };
    return absl::c_all_of(contracting_dims, is_unique) &&
           absl::c_all_of(batch_dims, is_unique);
  };

  if (!dims_unique(lhs_contracting_dimensions, lhs_batch_dimensions) ||
      !dims_unique(rhs_contracting_dimensions, rhs_batch_dimensions)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("A dimension number is not unique in Dot: %s.",
                        dimension_numbers.DebugString()));
  }

  return absl::OkStatus();
}

absl::Status CheckDotDimensionConstraints(
    const Shape& lhs, const Shape& rhs,
    const DotDimensionNumbers& dimension_numbers,
    std::optional<std::array<std::pair<int, int>, HloDotInstruction::kOperands>>
        sparsity_nm = std::nullopt,
    std::optional<std::array<int, HloDotInstruction::kOperands>> sparsity_dim =
        std::nullopt) {
  auto fail = [lhs, rhs](const std::string& addendum) -> absl::Status {
    std::string message = absl::StrFormat(
        "Cannot infer shape for dot operation: %s <dot> %s.",
        ShapeUtil::HumanString(lhs), ShapeUtil::HumanString(rhs));
    if (!addendum.empty()) {
      message += " " + addendum;
    }
    return absl::InvalidArgumentError(message);
  };

  // Check that number of contracting dimensions match.
  if (dimension_numbers.lhs_contracting_dimensions_size() !=
      dimension_numbers.rhs_contracting_dimensions_size()) {
    return fail(
        "Must specify the same number of contracting dimensions for lhs and "
        "rhs.");
  }
  // Check that contracting dimension sizes match.
  for (int64_t i = 0; i < dimension_numbers.lhs_contracting_dimensions_size();
       ++i) {
    const int64_t lhs_contracting_dimension =
        dimension_numbers.lhs_contracting_dimensions(i);
    const int64_t rhs_contracting_dimension =
        dimension_numbers.rhs_contracting_dimensions(i);
    int64_t lhs_size = lhs.dimensions(lhs_contracting_dimension);
    int64_t rhs_size = rhs.dimensions(rhs_contracting_dimension);
    bool is_sparse = false;
    if (sparsity_nm.has_value() && sparsity_dim.has_value()) {
      if (lhs_contracting_dimension == sparsity_dim.value()[0]) {
        lhs_size *= sparsity_nm.value()[0].second;
        rhs_size *= sparsity_nm.value()[0].first;
        is_sparse = true;
      }
      if (rhs_contracting_dimension == sparsity_dim.value()[1]) {
        lhs_size *= sparsity_nm.value()[1].first;
        rhs_size *= sparsity_nm.value()[1].second;
        is_sparse = true;
      }
    }
    if (!CompatibleDimensionSizes(lhs_size, rhs_size)) {
      return fail(
          !is_sparse
              ? "Contracting dimension sizes are not compatible."
              : "Sparse dimension size ratio doesn't match the descriptor.");
    }
  }

  // Check that number of batch dimensions match.
  if (dimension_numbers.lhs_batch_dimensions_size() !=
      dimension_numbers.rhs_batch_dimensions_size()) {
    return fail("Must the same number of batch dimensions for lhs and rhs.");
  }

  // Check that batch dimension numbers and sizes match.
  for (int64_t i = 0; i < dimension_numbers.lhs_batch_dimensions_size(); ++i) {
    if (!CompatibleDimensionSizes(
            lhs.dimensions(dimension_numbers.lhs_batch_dimensions(i)),
            rhs.dimensions(dimension_numbers.rhs_batch_dimensions(i)))) {
      return fail("Batch dimension sizes are not compatible.");
    }
  }
  return absl::OkStatus();
}

// The ranks of lhs and rhs are decremented by 1 respectively due to the
// contraction, and added for the rank of the result. When an input tensor is
// a scalar, its contribution to the rank of the result is 0.
// Generate the result dimensions in order, rhs dimensions followed by lhs
// dimensions except the contracted and batch dimensions.
void GenerateDotResultDimensions(
    const Shape& lhs, const Shape& rhs,
    const DotDimensionNumbers& dimension_numbers,
    std::vector<int64_t>& dimensions, std::vector<bool>& is_dynamic,
    std::vector<int64_t> rhs_group_dimensions = {}) {
  const auto& lhs_batch_dimensions = dimension_numbers.lhs_batch_dimensions();
  const auto lhs_batch_dimensions_size =
      lhs.rank() - dimension_numbers.lhs_contracting_dimensions().size() +
      rhs.rank() - dimension_numbers.rhs_contracting_dimensions().size() -
      dimension_numbers.rhs_batch_dimensions().size();
  dimensions.reserve(lhs_batch_dimensions_size);
  is_dynamic.reserve(lhs_batch_dimensions_size);
  for (const int64_t lhs_dim : lhs_batch_dimensions) {
    dimensions.push_back(lhs.dimensions(lhs_dim));
    is_dynamic.push_back(lhs.is_dynamic_dimension(lhs_dim));
  }
  for (int64_t i = 0; i < lhs.rank(); i++) {
    if (!absl::c_linear_search(dimension_numbers.lhs_contracting_dimensions(),
                               i) &&
        !absl::c_linear_search(dimension_numbers.lhs_batch_dimensions(), i)) {
      dimensions.push_back(lhs.dimensions(i));
      is_dynamic.push_back(lhs.is_dynamic_dimension(i));
    }
  }
  for (int64_t i = 0; i < rhs.rank(); i++) {
    if (!absl::c_linear_search(dimension_numbers.rhs_contracting_dimensions(),
                               i) &&
        !absl::c_linear_search(dimension_numbers.rhs_batch_dimensions(), i) &&
        !absl::c_linear_search(rhs_group_dimensions, i)) {
      dimensions.push_back(rhs.dimensions(i));
      is_dynamic.push_back(rhs.is_dynamic_dimension(i));
    }
  }
}

}  // namespace

// static
absl::StatusOr<Shape> ShapeInference::InferDotOpShape(
    const Shape& lhs, const Shape& rhs,
    const DotDimensionNumbers& dimension_numbers,
    std::optional<PrimitiveType> preferred_element_type,
    absl::Span<const SparsityDescriptor> sparsity) {
  TF_RETURN_IF_ERROR(ExpectArray(lhs, "lhs of dot"));
  TF_RETURN_IF_ERROR(ExpectArray(rhs, "rhs of dot"));

  // Validate basic properties of dot dimension numbers.
  TF_RETURN_IF_ERROR(ValidateDotDimensionNumbers(lhs, rhs, dimension_numbers));

  // Sparsity is only supported for contracting dimensions.
  // With N:M sparsity, the contracting dimension sizes have N/M ratio.
  const int kSize = HloDotInstruction::kOperands;
  std::array<std::pair<int, int>, kSize> sparsity_nm = {{{1, 1}, {1, 1}}};
  std::array<int, kSize> sparsity_dim = {-1, -1};
  for (const auto& descriptor : sparsity) {
    TF_RET_CHECK(descriptor.index() == 0 || descriptor.index() == 1);
    sparsity_dim[descriptor.index()] = descriptor.dimension();
    switch (descriptor.type()) {
      case SPARSITY_STRUCTURED_N_M:
        sparsity_nm[descriptor.index()] = {descriptor.n(), descriptor.m()};
        break;
      default:
        LOG(FATAL) << "Unsupported sparsity type: " << descriptor.type();
    }
  }

  // Check the number and sizes of batch and contracting dimensions.
  TF_RETURN_IF_ERROR(CheckDotDimensionConstraints(lhs, rhs, dimension_numbers,
                                                  sparsity_nm, sparsity_dim));

  std::vector<int64_t> dimensions;
  std::vector<bool> is_dynamic;
  GenerateDotResultDimensions(lhs, rhs, dimension_numbers, dimensions,
                              is_dynamic);

  PrimitiveType type = preferred_element_type.value_or(
      ShapeUtil::HigherPrecisionElementType(lhs, rhs));
  Shape result = ShapeUtil::MakeShape(type, dimensions, is_dynamic);

  DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(result));
  VLOG(2) << "inferred dot shape: " << ShapeUtil::HumanString(result);
  return result;
}

// static
absl::StatusOr<Shape> ShapeInference::InferDegenerateDimensionBroadcastShape(
    const Shape& lhs, const Shape& rhs) {
  TF_RET_CHECK(lhs.rank() == rhs.rank());

  // The shapes have to be compatible. That is, if some dimension d has a
  // different size in the two shapes, one of them has to be 1 (a "degenerate"
  // dimension). In that case, the output shape has the non-1 dimension size
  // from the lhs/rhs pair in every index.
  std::vector<int64_t> output_dimensions(lhs.rank());
  std::vector<bool> output_dimensions_is_dynamic(lhs.rank());
  for (int64_t i = 0; i < lhs.rank(); ++i) {
    if (lhs.dimensions(i) == 1 || rhs.dimensions(i) == 1) {
      // For the unbounded case, the operand with 1 should be broadcasted to the
      // unbounded size which can be > 1.
      // LHS | RHS | Result
      // 1   | X   | X
      // 1   | <=X | <=X
      // 1   | ?   | ?
      // X   | 1   | X
      // <=X | 1   | <=X
      // ?   | 1   | ?
      output_dimensions[i] =
          lhs.dimensions(i) == 1 ? rhs.dimensions(i) : lhs.dimensions(i);
      output_dimensions_is_dynamic[i] = lhs.dimensions(i) == 1
                                            ? rhs.is_dynamic_dimension(i)
                                            : lhs.is_dynamic_dimension(i);
    } else if (lhs.dimensions(i) == rhs.dimensions(i)) {
      // LHS | RHS | Result
      // X   | X   | X
      // X   | <=X | <=X
      // <=X | X   | <=X
      // <=X | <=X | <=X
      // ?   | ?   | ?
      output_dimensions[i] = lhs.dimensions(i);
      output_dimensions_is_dynamic[i] =
          lhs.is_dynamic_dimension(i) || rhs.is_dynamic_dimension(i);
    } else if (lhs.is_unbounded_dynamic_dimension(i) ||
               rhs.is_unbounded_dynamic_dimension(i)) {
      // For the last two rows, consider when <=X turns out to be 1 and ? turns
      // out to be 5. It would be wrong to infer <=1 as this is a degenerate
      // dimension that should be broadcasted to 5.
      // LHS | RHS | Result
      // X   | ?   | X
      // ?   | X   | X
      // <=X | ?   | ?
      // ?   | <=X | ?
      output_dimensions[i] = lhs.is_unbounded_dynamic_dimension(i)
                                 ? rhs.dimensions(i)
                                 : lhs.dimensions(i);
      output_dimensions_is_dynamic[i] = lhs.is_unbounded_dynamic_dimension(i)
                                            ? rhs.is_dynamic_dimension(i)
                                            : lhs.is_dynamic_dimension(i);
    } else {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Binary op with incompatible shapes: %s and %s.",
          ShapeUtil::HumanString(lhs), ShapeUtil::HumanString(rhs)));
    }
  }

  return ShapeUtil::MakeShape(ShapeUtil::HigherPrecisionElementType(lhs, rhs),
                              output_dimensions, output_dimensions_is_dynamic);
}

// static
absl::StatusOr<Shape> ShapeInference::InferInDimBroadcastShape(
    const Shape& smaller_shape, const Shape& larger_shape,
    absl::Span<const int64_t> broadcast_dimensions) {
  if (broadcast_dimensions.empty() && !ShapeUtil::IsScalar(smaller_shape)) {
    // Reject "magic" inference for binops on different shapes, requiring
    // the user to provide an explicit broadcast dimension in this case.
    // See b/25177275 for more details.
    return absl::InvalidArgumentError(
        absl::StrFormat("Shapes must be equal rank, but are %s and %s",
                        ShapeUtil::HumanString(smaller_shape),
                        ShapeUtil::HumanString(larger_shape)));
  }

  if (broadcast_dimensions.size() != smaller_shape.rank()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Size of broadcast_dimensions has to match lower-rank operand's "
        "rank; "
        " lower-rank operand's rank is %d, size of broadcast_dimensions is "
        "%u.",
        smaller_shape.rank(), broadcast_dimensions.size()));
  }

  // broadcast_dimensions is a sequence of dimensions; its length is equal to
  // the rank of the lower-rank operand. The lower-rank operand's dimensions
  // have to be compatible with the higher-rank operand's dimensions at indices
  // specified by broadcast_dimensions. Here compatible means the dimension
  // sizes are equal or in one of the shapes the dimension size is
  // one. Examples:
  //
  // smaller_shape   larger_shape   broadcast_dimensions   output_shape
  //   []              [2, 3]          {}                    [2, 3]
  //   [3]             [4, 3]          {1}                   [4, 3]
  //   [2, 3]          [2, 3, 4]       {0, 1}                [2, 3, 4]
  //   [2, 1]          [2, 3, 4]       {0, 2}                [2, 3, 1]
  //   [2, 3]          [2, 1, 4]       {0, 1}                [2, 3, 4]
  //
  // The column output_shape may not be the final shape of the ZKX
  // operation. After the "InDim" broadcasting implemented in this function
  // expands the rank, degenerate-dimension broadcasting (implemented in
  // InferDegenerateDimensionBroadcastShape) broadcasts dimensions of size one
  // up to match the dimension size of the other operand. For example, consider
  // the row in the table above with a smaller_shape of [2, 1]. The shape
  // returned by this function is [2, 3, 1] (output_shape) however, the result
  // shape of the ZKX operation is [2, 3, 4] after degenerate-dimension
  // broadcasting.
  //
  // Invalid broadcasts:
  //
  // smaller_shape=[3], larger_shape=[4, 3], broadcast_dimensions={0}
  // Reason: Dimension zero** of larger_shape (size 4) is not compatible with
  //   dimension zero of smaller_shape(size 3). **Zero here comes from the value
  //   in broadcast_dimensions.
  //
  // smaller_shape=[2, 1], larger_shape=[2, 3, 4], broadcast_dimensions={1, 2}
  // Reason: Dimension one of larger_shape (size 3) is not compatible with
  //   dimension zero of smaller_shape(size 2)

  // The output shape is initially the larger_shape. Sizes of dimensions
  // specified in broadcast_dimensions are then changed to match the
  // corresponding dimension size in smaller_shape.
  Shape output_shape(larger_shape);
  output_shape.set_element_type(
      ShapeUtil::HigherPrecisionElementType(larger_shape, smaller_shape));

  for (int i = 0; i < smaller_shape.dimensions_size(); ++i) {
    int64_t dimension_to_match = broadcast_dimensions.at(i);
    if (dimension_to_match < 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Broadcast dimension number (%d) cannot be negative.",
                          dimension_to_match));
    }
    if (dimension_to_match >= larger_shape.dimensions_size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Broadcast dimension number (%d) too large; higher-rank "
          "operand has rank %d.",
          dimension_to_match, larger_shape.dimensions_size()));
    }
    int64_t small_dimension_size = smaller_shape.dimensions(i);
    int64_t large_dimension_size = larger_shape.dimensions(dimension_to_match);
    bool small_is_dynamic = smaller_shape.is_dynamic_dimension(i);
    bool large_is_dynamic =
        larger_shape.is_dynamic_dimension(dimension_to_match);
    // Dimension sizes must be compatible: match or be degenerate (degenerate
    // case is handled by degenerate dimension broadcasting which occurs after
    // InDim broadcasting).
    if (small_dimension_size != large_dimension_size &&
        small_dimension_size != 1 && large_dimension_size != 1) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Broadcast dimension %d mismatch: %d != %d; %s and %s.", i,
          small_dimension_size, large_dimension_size,
          ShapeUtil::HumanString(smaller_shape),
          ShapeUtil::HumanString(larger_shape)));
    }
    if (small_is_dynamic != large_is_dynamic) {
      if (small_dimension_size == large_dimension_size ||
          (small_dimension_size == 1 && !small_is_dynamic) ||
          (large_dimension_size == 1 && !large_is_dynamic)) {
        // Do nothing. It's OK when the size-1 dimension is not static or when
        // it is unbounded dynamic.
      } else {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Broadcast dimension %d dynamism mismatch: %s and %s.", i,
            ShapeUtil::HumanString(smaller_shape),
            ShapeUtil::HumanString(larger_shape)));
      }
    }
    // Make sure the broadcast dimensions are listed in a strictly increasing
    // order.
    if (i > 0 && broadcast_dimensions.at(i - 1) >= dimension_to_match) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Broadcast dimensions order is wrong: %d comes after %d.",
          dimension_to_match, broadcast_dimensions.at(i - 1)));
    }

    output_shape.set_dimensions(dimension_to_match, small_dimension_size);
    output_shape.set_dynamic_dimension(dimension_to_match, small_is_dynamic);
  }

  return output_shape;
}

// static
absl::StatusOr<Shape> ShapeInference::InferElementwiseBinaryOpShape(
    HloOpcode operation, const Shape& lhs, const Shape& rhs,
    absl::Span<const int64_t> broadcast_dimensions) {
  TF_RETURN_IF_ERROR(ExpectArray(lhs, "lhs of elementwise binary operation"));
  TF_RETURN_IF_ERROR(ExpectArray(rhs, "rhs of elementwise binary operation"));

  if (!ShapeUtil::SameElementType(lhs, rhs)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Binary op %s with different element types: %s and %s.",
                        HloOpcodeString(operation), ShapeUtil::HumanString(lhs),
                        ShapeUtil::HumanString(rhs)));
  }

  if (lhs.rank() == rhs.rank()) {
    std::vector<int64_t> identity_dims(lhs.rank());
    std::iota(identity_dims.begin(), identity_dims.end(), 0);
    if (!broadcast_dimensions.empty() &&
        broadcast_dimensions != identity_dims) {
      return absl::InvalidArgumentError(
          "Broadcast dimensions field must either be not set or be the "
          "identity on binary operations with operands of the same rank.");
    }
  }

  if (ShapeUtil::Compatible(lhs, rhs) && !lhs.is_unbounded_dynamic() &&
      !rhs.is_unbounded_dynamic()) {
    // If the shapes are the same other than layout, the output shape is the
    // same (elementwise op).
    Shape result = ShapeUtil::ChangeElementType(
        lhs, ShapeUtil::HigherPrecisionElementType(lhs, rhs));

    for (int64_t i = 0; i < rhs.rank(); ++i) {
      if (rhs.is_dynamic_dimension(i)) {
        result.set_dynamic_dimension(i, true);
      }
    }

    return result;

  } else if (lhs.rank() == rhs.rank()) {
    return InferDegenerateDimensionBroadcastShape(lhs, rhs);
  } else {
    // Ranks do not match, so perform InDim broadcasting using
    // broadcast_dimensions. Scalar broadcasting is a special case of this.
    const Shape& larger_shape = lhs.rank() > rhs.rank() ? lhs : rhs;
    const Shape& smaller_shape = lhs.rank() > rhs.rank() ? rhs : lhs;

    // After InDim broadcasting, perform degenerate dimensions broadcasting.
    TF_ASSIGN_OR_RETURN(Shape indim_broadcast_shape,
                        InferInDimBroadcastShape(smaller_shape, larger_shape,
                                                 broadcast_dimensions));

    return InferDegenerateDimensionBroadcastShape(indim_broadcast_shape,
                                                  larger_shape);
  }
}

// static
absl::StatusOr<Shape> ShapeInference::InferBinaryOpShape(
    HloOpcode opcode, const HloInstruction* lhs, const HloInstruction* rhs) {
  return InferBinaryOpShape(opcode, lhs->shape(), rhs->shape(),
                            /*broadcast_dimensions=*/{});
}

// static
absl::StatusOr<Shape> ShapeInference::InferBinaryOpShape(
    HloOpcode opcode, const Shape& lhs, const Shape& rhs,
    absl::Span<const int64_t> broadcast_dimensions) {
  VLOG(2) << absl::StrFormat(
      "inferring shape for <%s>(%s, %s) with broadcast_dimensions={%s}",
      HloOpcodeString(opcode), ShapeUtil::HumanStringWithLayout(lhs),
      ShapeUtil::HumanStringWithLayout(rhs),
      absl::StrJoin(broadcast_dimensions, ", "));

  DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(lhs));
  DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(rhs));

  TF_RETURN_IF_ERROR(ExpectArray(
      lhs, absl::StrCat("lhs of binary operation ", HloOpcodeString(opcode))));
  TF_RETURN_IF_ERROR(ExpectArray(
      rhs, absl::StrCat("rhs of binary operation ", HloOpcodeString(opcode))));
  switch (opcode) {
    case HloOpcode::kAdd:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
      return InferElementwiseBinaryOpShape(opcode, lhs, rhs,
                                           broadcast_dimensions);

    case HloOpcode::kSubtract:
    case HloOpcode::kPower:
    case HloOpcode::kDivide:
      if (lhs.element_type() == PRED || rhs.element_type() == PRED) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Expected element type in shape to be arithmetic type for "
            "operation %s; got PRED.",
            HloOpcodeString(opcode)));
      }
      return InferElementwiseBinaryOpShape(opcode, lhs, rhs,
                                           broadcast_dimensions);

      // clang-format off
    // TODO(chokobole): Uncomment this. Dependency: HloOpcode::kAnd, HloOpcode::kOr, HloOpcode::kXor
    // clang-format on
    // case HloOpcode::kAnd:
    // case HloOpcode::kOr:
    // case HloOpcode::kXor: {
    case HloOpcode::kCompare: {
      TF_ASSIGN_OR_RETURN(const Shape& shape,
                          InferElementwiseBinaryOpShape(opcode, lhs, rhs,
                                                        broadcast_dimensions));
      return ShapeUtil::ChangeElementType(shape, PRED);
    }
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Binary op shape inference: %s; lhs: %s; rhs: %s is not implemented.",
          HloOpcodeString(opcode), lhs.ShortDebugString(),
          rhs.ShortDebugString()));
  }
}

// static
absl::StatusOr<Shape> ShapeInference::InferTernaryOpShape(
    HloOpcode opcode, const HloInstruction* lhs, const HloInstruction* rhs,
    const HloInstruction* ehs) {
  return InferTernaryOpShape(opcode, lhs->shape(), rhs->shape(), ehs->shape());
}

// static
absl::StatusOr<Shape> ShapeInference::InferTernaryOpShape(HloOpcode opcode,
                                                          const Shape& lhs,
                                                          const Shape& rhs,
                                                          const Shape& ehs) {
  DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(lhs));
  DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(rhs));
  DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(ehs));
  switch (opcode) {
    case HloOpcode::kSelect:
      return InferSelectShape(lhs, rhs, ehs);
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unknown operation %s.", HloOpcodeString(opcode)));
  }
}

// static
absl::StatusOr<Shape> ShapeInference::InferVariadicOpShape(
    HloOpcode opcode, absl::Span<const HloInstruction* const> operands) {
  std::vector<const Shape*> operand_shapes;
  operand_shapes.reserve(operands.size());
  for (const HloInstruction* operand : operands) {
    operand_shapes.push_back(&operand->shape());
  }
  return InferVariadicOpShape(opcode, operand_shapes);
}

// static
absl::StatusOr<Shape> ShapeInference::InferVariadicOpShape(
    HloOpcode opcode, absl::Span<const Shape* const> operand_shapes) {
  for (const Shape* shape : operand_shapes) {
    DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(*shape));
  }
  switch (opcode) {
    case HloOpcode::kTuple: {
      Shape result = ShapeUtil::MakeTupleShape({});
      result.mutable_tuple_shapes()->reserve(operand_shapes.size());
      for (const Shape* shape : operand_shapes) {
        ShapeUtil::AppendShapeToTuple(*shape, &result);
      }
      return result;
    }
    // TODO(chokobole): Uncomment this. Dependency HloOpcode::kSort
    // case HloOpcode::kSort: {
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unknown operation %s.", HloOpcodeString(opcode)));
  }
}

// static
absl::StatusOr<Shape> ShapeInference::InferFftShape(const Shape& in,
                                                    FftType fft_type) {
  switch (fft_type) {
    case FFT:
    case IFFT:
      return in;
    default:
      LOG(FATAL) << "Unexpected fft_type: " << fft_type;
  }
}

// static
absl::StatusOr<Shape> ShapeInference::InferMsmShape(const Shape& bases) {
  return ShapeUtil::MakeScalarShape(bases.element_type());
}

// static
absl::StatusOr<Shape> ShapeInference::InferAllGatherShape(
    absl::Span<const Shape* const> operand_shapes, int64_t all_gather_dimension,
    int64_t shard_count) {
  TF_RET_CHECK(all_gather_dimension >= 0);
  TF_RET_CHECK(shard_count > 0);

  std::vector<Shape> output_shapes;
  output_shapes.reserve(operand_shapes.size());
  for (const Shape* operand_shape : operand_shapes) {
    TF_RET_CHECK(all_gather_dimension < operand_shape->rank());
    TF_RETURN_IF_ERROR(ExpectArray(*operand_shape, "operand of all-gather"));

    Shape output_shape = *operand_shape;
    int64_t output_shape_dimension =
        output_shape.dimensions(all_gather_dimension);
    output_shape.set_dimensions(all_gather_dimension,
                                IsUnboundedDynamicSize(output_shape_dimension)
                                    ? Shape::kUnboundedSize
                                    : shard_count * output_shape_dimension);
    output_shapes.push_back(output_shape);
  }
  if (output_shapes.size() == 1) {
    return output_shapes[0];
  }
  return ShapeUtil::MakeTupleShape(output_shapes);
}

// static
absl::StatusOr<Shape> ShapeInference::InferAllGatherStartShape(
    absl::Span<const Shape* const> operand_shapes, int64_t all_gather_dimension,
    int64_t shard_count) {
  TF_ASSIGN_OR_RETURN(
      Shape ag_shape,
      InferAllGatherShape(operand_shapes, all_gather_dimension, shard_count));
  Shape input_shape;
  if (operand_shapes.size() == 1) {
    input_shape = *operand_shapes[0];
  } else {
    input_shape = ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes);
  }
  return ShapeUtil::MakeTupleShapeWithPtrs({&input_shape, &ag_shape});
}

// static
absl::StatusOr<Shape> ShapeInference::InferAllGatherDoneShape(
    const Shape& all_gather_start_shape) {
  return ShapeUtil::GetTupleElementShape(all_gather_start_shape, 1);
}

// static
absl::StatusOr<Shape> ShapeInference::InferAllReduceShape(
    absl::Span<const Shape* const> operand_shapes) {
  for (const Shape* operand_shape : operand_shapes) {
    TF_RETURN_IF_ERROR(
        ExpectArray(*operand_shape, "operand of cross replica sum"));
  }
  if (operand_shapes.size() == 1) {
    return *operand_shapes[0];
  }
  return ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes);
}

// static
absl::StatusOr<Shape> ShapeInference::InferReduceScatterShape(
    absl::Span<const Shape* const> operand_shapes, int64_t scatter_dimension,
    int64_t shard_count) {
  TF_RET_CHECK(scatter_dimension >= 0);
  TF_RET_CHECK(shard_count > 0);

  std::vector<Shape> output_shapes;
  output_shapes.reserve(operand_shapes.size());
  for (const Shape* operand_shape : operand_shapes) {
    TF_RET_CHECK(scatter_dimension < operand_shape->rank());
    TF_RETURN_IF_ERROR(
        ExpectArray(*operand_shape, "operand of reduce-scatter"));

    int64_t scatter_dim_input_size =
        operand_shape->dimensions(scatter_dimension);
    if (scatter_dim_input_size % shard_count != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "ReduceScatter operand scatter dimension size %d must be "
          "dividable by shard_count "
          "%d.",
          scatter_dim_input_size, shard_count));
    }

    Shape output_shape = *operand_shape;
    output_shape.set_dimensions(
        scatter_dimension, output_shape.is_dynamic_dimension(scatter_dimension)
                               ? Shape::kUnboundedSize
                               : scatter_dim_input_size / shard_count);
    output_shapes.push_back(output_shape);
  }

  if (output_shapes.size() == 1) {
    return output_shapes[0];
  }
  return ShapeUtil::MakeTupleShape(output_shapes);
}

// static
absl::StatusOr<Shape> ShapeInference::InferAllReduceStartShape(
    absl::Span<const Shape* const> operand_shapes) {
  return InferAllReduceShape(operand_shapes);
}

// static
absl::StatusOr<Shape> ShapeInference::InferAllReduceDoneShape(
    const Shape& operand_shape) {
  // The returned value from AllReduceDone is the operand forwarded.
  return operand_shape;
}

// static
absl::StatusOr<Shape> ShapeInference::InferAllToAllShape(
    const Shape& shape, int64_t split_dimension, int64_t concat_dimension,
    int64_t split_count) {
  TF_RET_CHECK(split_count > 0);
  TF_RET_CHECK(!shape.is_bounded_dynamic())
      << "AllToAll does not support bounded dynamic shapes";
  if (split_dimension >= shape.rank() || split_dimension < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "AllToAll split_dimension %d is out-of-bounds in shape %s.",
        split_dimension, ShapeUtil::HumanString(shape)));
  }
  if (concat_dimension >= shape.rank() || concat_dimension < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "AllToAll concat_dimension %d is out-of-bounds in shape %s.",
        concat_dimension, ShapeUtil::HumanString(shape)));
  }
  int64_t split_dimension_size = shape.dimensions(split_dimension);
  if (!IsUnboundedDynamicSize(split_dimension_size) &&
      split_dimension_size % split_count != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "AllToAll split dimension size %d must be dividable by split_count "
        "%d.",
        split_dimension_size, split_count));
  }
  std::vector<int64_t> new_dimensions(shape.dimensions().begin(),
                                      shape.dimensions().end());
  new_dimensions[split_dimension] =
      IsUnboundedDynamicSize(new_dimensions[split_dimension])
          ? Shape::kUnboundedSize
          : new_dimensions[split_dimension] / split_count;
  new_dimensions[concat_dimension] =
      IsUnboundedDynamicSize(new_dimensions[concat_dimension])
          ? Shape::kUnboundedSize
          : new_dimensions[concat_dimension] * split_count;

  const std::vector<bool> dynamic_dimensions(shape.dynamic_dimensions().begin(),
                                             shape.dynamic_dimensions().end());
  return ShapeUtil::MakeShape(shape.element_type(), new_dimensions,
                              dynamic_dimensions);
}

// static
absl::StatusOr<Shape> ShapeInference::InferAllToAllTupleShape(
    absl::Span<const Shape* const> operand_shapes) {
  // An AllToAll HLO instruction receives N operands (with the same shape) and
  // returns a tuple that contains N array shapes.
  TF_RET_CHECK(!operand_shapes.empty());
  for (int i = 0; i < operand_shapes.size(); i++) {
    if (operand_shapes[i]->is_unbounded_dynamic()) {
      return absl::InvalidArgumentError(
          "AllToAllTuple does not support unbounded dynamic shapes");
    }
    if (!Shape::Equal().IgnoreMemorySpaceInLayout()(*operand_shapes[0],
                                                    *operand_shapes[i])) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "HLO all-to-all has operands with different shapes: the 0th "
          "operand shape %s, but the %dth operand has shape %s.",
          ShapeUtil::HumanString(*operand_shapes[0]), i,
          ShapeUtil::HumanString(*operand_shapes[i])));
    }
  }

  return InferVariadicOpShape(HloOpcode::kTuple, operand_shapes);
}

// static
absl::StatusOr<Shape> ShapeInference::InferRaggedAllToAllShape(
    absl::Span<const Shape* const> operand_shapes) {
  TF_RETURN_IF_ERROR(
      ExpectArray(*(operand_shapes[1]), "operand 1 of ragged-all-to-all"));
  return *(operand_shapes[1]);
}

// static
absl::StatusOr<Shape> ShapeInference::InferCollectiveBroadcastShape(
    absl::Span<const Shape* const> operand_shapes) {
  TF_RETURN_IF_ERROR(
      ExpectArray(*(operand_shapes[0]), "operand of collective-broadcast"));
  return *(operand_shapes[0]);
}

// static
absl::StatusOr<Shape> ShapeInference::InferCollectivePermuteShape(
    absl::Span<const Shape* const> operand_shapes, bool inplace) {
  if (!inplace) {
    for (const Shape* operand_shape : operand_shapes) {
      TF_RETURN_IF_ERROR(
          ExpectArray(*operand_shape, "operand of collective-permute"));
    }
    if (operand_shapes.size() == 1) {
      return *operand_shapes[0];
    }
    return ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes);
  } else {
    TF_RET_CHECK(operand_shapes.size() == 4);
    return *(operand_shapes[1]);
  }
}

// static
absl::StatusOr<Shape> ShapeInference::InferCollectivePermuteStartShape(
    absl::Span<const Shape* const> operand_shapes,
    absl::Span<const Shape> context_shapes, bool inplace) {
  absl::InlinedVector<Shape, 4> shapes;
  if (!inplace) {
    if (operand_shapes.size() == 1) {
      TF_RETURN_IF_ERROR(ExpectArray(*(operand_shapes[0]),
                                     "operand of collective-permute-start"));
      shapes = {*operand_shapes[0], *operand_shapes[0]};
    } else {
      Shape tuple_shape = ShapeUtil::MakeTupleShapeWithPtrs(operand_shapes);
      shapes = {tuple_shape, tuple_shape};
    }
  } else {
    TF_RET_CHECK(operand_shapes.size() == 4);
    shapes = {*operand_shapes[0], *operand_shapes[1]};
  }
  absl::c_transform(context_shapes, std::back_inserter(shapes),
                    [](const Shape& shape) { return shape; });
  return ShapeUtil::MakeTupleShape(shapes);
}

// static
absl::StatusOr<Shape> ShapeInference::InferCollectivePermuteDoneShape(
    const Shape& operand_shape) {
  TF_RET_CHECK(operand_shape.IsTuple());
  return ShapeUtil::GetTupleElementShape(operand_shape, 1);
}

// static
absl::StatusOr<Shape> ShapeInference::InferBroadcastShape(
    const Shape& operand, absl::Span<const int64_t> broadcast_sizes) {
  // This method is used to infer shape for zkx::BroadcastInDim.
  TF_RETURN_IF_ERROR(ExpectArray(operand, "operand of broadcast"));
  TF_RET_CHECK(!operand.is_unbounded_dynamic());
  for (int64_t size : broadcast_sizes) {
    if (size == Shape::kUnboundedSize) {
      return absl::InvalidArgumentError(
          "Non-broadcast dimensions must not be dynamic.");
    }
    if (size < 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Broadcast with negative dimension size %d.", size));
    }
  }

  std::vector<int64_t> dimensions(operand.dimensions_size() +
                                  broadcast_sizes.size());
  std::copy(broadcast_sizes.begin(), broadcast_sizes.end(), dimensions.begin());
  std::copy(operand.dimensions().begin(), operand.dimensions().end(),
            dimensions.begin() + broadcast_sizes.size());

  TF_ASSIGN_OR_RETURN(Shape result, ShapeUtil::MakeValidatedShape(
                                        operand.element_type(), dimensions));
  for (int64_t i = 0; i < operand.dimensions_size(); ++i) {
    result.set_dynamic_dimension(broadcast_sizes.size() + i,
                                 operand.is_dynamic_dimension(i));
  }
  return result;
}

absl::StatusOr<Shape> ShapeInference::InferBroadcastShape(
    const Shape& operand_shape, const Shape& output_shape,
    absl::Span<const int64_t> broadcast_dimensions) {
  // This method is used to infer shape for zkx::BroadcastInDim.
  TF_RETURN_IF_ERROR(ExpectArray(operand_shape, "operand of broadcast"));
  TF_RETURN_IF_ERROR(ExpectArray(output_shape, "operand of broadcast"));
  TF_RET_CHECK(!output_shape.is_unbounded_dynamic());
  const int64_t operand_rank = operand_shape.rank();
  const int64_t output_rank = output_shape.rank();
  if (operand_rank > output_rank) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "InDim style broadcast must be to an equal or higher ranked shape; "
        "operand rank: %lld; output rank: %lld",
        operand_rank, output_rank));
  }
  if (operand_rank != broadcast_dimensions.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Size of broadcast_dimensions has to match operand's rank; operand "
        "rank: %lld, size of broadcast_dimensions %u.",
        operand_rank, broadcast_dimensions.size()));
  }
  for (int64_t i = 0; i < operand_rank; i++) {
    if (broadcast_dimensions[i] < 0 || broadcast_dimensions[i] >= output_rank) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Broadcast dimension %lld is out of bound", broadcast_dimensions[i]));
    }
    if (!operand_shape.is_unbounded_dynamic_dimension(i) &&
        operand_shape.dimensions(i) !=
            output_shape.dimensions(broadcast_dimensions[i]) &&
        operand_shape.dimensions(i) != 1) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Input dimension should be either 1 or equal to the output dimension "
          "it is broadcasting into; the %lldth operand dimension is %lld, the "
          "%lldth output dimension is %lld.",
          i, operand_shape.dimensions(i), broadcast_dimensions[i],
          output_shape.dimensions(broadcast_dimensions[i])));
    }
    if (!operand_shape.is_unbounded_dynamic_dimension(i) &&
        operand_shape.is_bounded_dynamic_dimension(i) !=
            output_shape.is_bounded_dynamic_dimension(
                broadcast_dimensions[i])) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Broadcast input and output dynamism mismatch: %s and %s",
          operand_shape.ToString(), output_shape.ToString()));
    }
    // Make sure the broadcast dimensions are listed in a strictly increasing
    // order.
    if (i > 0 && broadcast_dimensions[i - 1] >= broadcast_dimensions[i]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Broadcast dimensions order is wrong: %d comes after %d.",
          broadcast_dimensions[i], broadcast_dimensions.at(i - 1)));
    }
  }

  return output_shape;
}

// static
absl::StatusOr<Shape> ShapeInference::InferSelectShape(const Shape& pred,
                                                       const Shape& on_true,
                                                       const Shape& on_false) {
  TF_RETURN_IF_ERROR(ExpectArray(pred, "select pred"));
  TF_RETURN_IF_ERROR(ExpectArray(on_true, "select on-true"));
  TF_RETURN_IF_ERROR(ExpectArray(on_false, "select on-false"));

  if (!ShapeUtil::Compatible(on_true, on_false)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Operands to select must be the same shape; got %s and %s.",
        ShapeUtil::HumanString(on_true), ShapeUtil::HumanString(on_false)));
  }

  if (pred.element_type() != PRED) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Select's pred operand must have PRED element type; got %s.",
        ShapeUtil::HumanString(pred)));
  }

  // If pred is not scalar, it must be compatible with on_true and on_false
  if ((!ShapeUtil::IsScalar(pred) &&
       (!ShapeUtil::CompatibleIgnoringElementType(pred, on_true) ||
        !ShapeUtil::CompatibleIgnoringElementType(pred, on_false))) ||
      !ShapeUtil::Compatible(on_true, on_false)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Operands to select and predicate must be the same shape; got %s and "
        "%s and %s.",
        ShapeUtil::HumanString(on_true), ShapeUtil::HumanString(on_false),
        ShapeUtil::HumanString(pred)));
  }

  Shape full_rank_shape = ShapeUtil::IsScalar(pred) ? on_true : pred;
  Shape result = ShapeUtil::ChangeElementType(
      full_rank_shape,
      ShapeUtil::HigherPrecisionElementType(on_true, on_false));
  for (int64_t dimension = 0; dimension < full_rank_shape.rank(); ++dimension) {
    if (on_true.is_unbounded_dynamic_dimension(dimension) ||
        on_false.is_unbounded_dynamic_dimension(dimension)) {
      absl::StatusOr<DimAndBound> inferred = InferMostSpecificDimAndBound(
          dimension, on_true.dimensions(dimension),
          on_false.dimensions(dimension), on_true.dimensions(dimension),
          on_false.dimensions(dimension));
      result.set_dimensions(dimension, (*inferred).dimension);
      result.set_dynamic_dimension(
          dimension, on_true.is_dynamic_dimension(dimension) &&
                         on_false.is_dynamic_dimension(dimension));
    } else {
      result.set_dynamic_dimension(
          dimension, (!ShapeUtil::IsScalar(pred) &&
                      pred.is_dynamic_dimension(dimension)) ||
                         on_true.is_dynamic_dimension(dimension) ||
                         on_false.is_dynamic_dimension(dimension));
    }
  }
  if (result.has_layout()) {
    result.mutable_layout()->set_element_size_in_bits(
        on_true.layout().element_size_in_bits());
  }
  return std::move(result);
}

}  // namespace zkx
