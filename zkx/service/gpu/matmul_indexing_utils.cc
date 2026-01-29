/* Copyright 2022 The OpenXLA Authors.
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

#include "zkx/service/gpu/matmul_indexing_utils.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/status_macros.h"
#include "zkx/util.h"

namespace zkx::gpu {

absl::StatusOr<std::vector<int64_t>> GetNonContractingDims(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contracting_dims) {
  auto nc =
      ::zkx::GetNonContractingDims(shape.rank(), contracting_dims, batch_dims);

  TF_RET_CHECK(batch_dims.size() + contracting_dims.size() + nc.size() ==
               shape.rank());
  return std::vector<int64_t>(nc.begin(), nc.end());
}

const google::protobuf::RepeatedField<int64_t>& BatchDimensionsForOperand(
    const HloInstruction& dot, int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    return dimension_numbers.lhs_batch_dimensions();
  }
  return dimension_numbers.rhs_batch_dimensions();
}

absl::StatusOr<int64_t> ContractingDimensionIndex(const HloInstruction& dot,
                                                  int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    TF_RET_CHECK(dimension_numbers.lhs_contracting_dimensions().size() == 1);
    return dimension_numbers.lhs_contracting_dimensions(0);
  }
  TF_RET_CHECK(dimension_numbers.rhs_contracting_dimensions().size() == 1);
  return dimension_numbers.rhs_contracting_dimensions(0);
}

absl::StatusOr<int64_t> NonContractingDimensionIndex(const HloInstruction& dot,
                                                     int operand_number) {
  TF_ASSIGN_OR_RETURN(int64_t contracting_dim,
                      ContractingDimensionIndex(dot, operand_number));
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> non_contracting_dims,
      GetNonContractingDims(dot.operand(operand_number)->shape(),
                            BatchDimensionsForOperand(dot, operand_number),
                            {contracting_dim}));
  TF_RET_CHECK(non_contracting_dims.size() == 1);
  return non_contracting_dims.front();
}

bool CanBatchRowColLayoutFor(const Shape& shape,
                             absl::Span<const int64_t> batch_dims,
                             absl::Span<const int64_t> row_dims,
                             absl::Span<const int64_t> col_dims) {
  if (!shape.has_layout()) {
    return false;
  }

  // This is a direct port of XLA's GetBatchRowColumnShape + MatrixLayout::For()
  // validation logic.
  //
  // Step 1: Check that batch, row, col dim groups are each laid out physically
  // sequentially (from GetBatchRowColumnShape).
  size_t i = 0;
  std::vector<int64_t> result_minor_to_major;
  for (; i < shape.rank();) {
    auto check_physically_sequential =
        [&](absl::Span<const int64_t> dims) -> bool {
      for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
        if (i >= shape.rank() || *it != shape.layout().minor_to_major()[i++]) {
          return false;
        }
      }
      return true;
    };

    int64_t dim = shape.layout().minor_to_major()[i];
    if (!row_dims.empty() && dim == row_dims.back()) {
      result_minor_to_major.push_back(1);  // row group marker
      if (!check_physically_sequential(row_dims)) {
        return false;
      }
    } else if (!col_dims.empty() && dim == col_dims.back()) {
      result_minor_to_major.push_back(2);  // col group marker
      if (!check_physically_sequential(col_dims)) {
        return false;
      }
    } else if (!batch_dims.empty() && dim == batch_dims.back()) {
      result_minor_to_major.push_back(0);  // batch group marker
      if (!check_physically_sequential(batch_dims)) {
        return false;
      }
    } else {
      return false;
    }
  }

  // Handle empty dimension groups (they still need positions in 3D layout).
  if (col_dims.empty()) result_minor_to_major.push_back(2);
  if (row_dims.empty()) result_minor_to_major.push_back(1);
  if (batch_dims.empty()) result_minor_to_major.push_back(0);

  // Step 2: Check that the resulting 3D layout is cuBLAS compatible
  // (from MatrixLayout::For(shape3d)).
  // The 3D shape has dimensions [batch, rows, cols] with indices [0, 1, 2].
  // Supported layouts (from XLA's MatrixLayout::For):
  //   012 (B,R,C major-to-minor), 021 (B,C,R), 0102 (R,B,C), 0201 (C,B,R)
  // NOT supported: batch in most minor dimension.
  if (result_minor_to_major.size() >= 3) {
    int code = 64 * result_minor_to_major[2] + 8 * result_minor_to_major[1] +
               result_minor_to_major[0];
    // Octal: 012=10, 021=17, 0102=66, 0201=129
    if (code != 10 && code != 17 && code != 66 && code != 129) {
      return false;  // batch in most minor dimension - not supported
    }
  }

  return true;
}

}  // namespace zkx::gpu
