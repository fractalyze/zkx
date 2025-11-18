/* Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_MATH_BASE_SPARSE_MATRIX_H_
#define ZKX_MATH_BASE_SPARSE_MATRIX_H_

#include <stdint.h>

#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/base/auto_reset.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/buffer/vector_buffer.h"
#include "zkx/base/random.h"

namespace zkx::math {

template <typename T>
class SparseMatrix {
 public:
  SparseMatrix() = default;
  ~SparseMatrix() = default;

  // Constructor with dimensions
  SparseMatrix(uint32_t num_rows, uint32_t num_cols)
      : num_rows_(num_rows), num_cols_(num_cols) {}

  SparseMatrix(uint32_t num_rows, uint32_t num_cols, uint32_t num_nonzeros)
      : num_rows_(num_rows), num_cols_(num_cols) {
    row_indices_.resize(num_nonzeros);
    col_indices_.resize(num_nonzeros);
    values_.resize(num_nonzeros);
  }

  // Generates a random sparse matrix with the given dimensions and number of
  // non-zero elements.
  static SparseMatrix Random(uint32_t num_rows, uint32_t num_cols,
                             uint32_t num_nonzeros) {
    SparseMatrix ret(num_rows, num_cols);

    struct Position {
      uint32_t row;
      uint32_t col;
    };

    // Generate all possible positions
    std::vector<Position> positions;
    positions.reserve(num_rows * num_cols);
    for (uint32_t i = 0; i < num_rows; ++i) {
      for (uint32_t j = 0; j < num_cols; ++j) {
        positions.push_back({i, j});
      }
    }

    base::Shuffle(positions);

    for (size_t i = 0; i < num_nonzeros; ++i) {
      if constexpr (std::is_integral_v<T>) {
        using UnsignedT =
            std::conditional_t<std::is_signed_v<T>, std::make_unsigned_t<T>, T>;
        ret.Insert(positions[i].row, positions[i].col,
                   absl::bit_cast<T>(base::Uniform<UnsignedT>()));
      } else {
        ret.Insert(positions[i].row, positions[i].col, T::Random());
      }
    }

    return ret;
  }

  // Get number of non-zero elements
  uint32_t NumNonZeros() const { return values_.size(); }

  // Get matrix dimensions
  uint32_t num_rows() const { return num_rows_; }
  uint32_t num_cols() const { return num_cols_; }

  // Returns a pointer to the value at (row, col).
  // Returns nullptr if the element is not present.
  // CHECK fails if row >= num_rows_ or col >= num_cols_.
  const T* operator()(uint32_t row, uint32_t col) const {
    CHECK_LT(row, num_rows_);
    CHECK_LT(col, num_cols_);
    for (uint32_t i = 0; i < row_indices_.size(); ++i) {
      if (row_indices_[i] == row && col_indices_[i] == col) {
        return &values_[i];
      }
    }
    return nullptr;
  }

  // Returns a pointer to the value at (row, col).
  // Returns nullptr if the element is not present.
  // CHECK fails if row >= num_rows_ or col >= num_cols_.
  T* operator()(uint32_t row, uint32_t col) {
    return const_cast<T*>(std::as_const(*this)(row, col));
  }

  bool operator==(const SparseMatrix& other) const {
    if (num_rows_ != other.num_rows_ || num_cols_ != other.num_cols_ ||
        values_.size() != other.values_.size()) {
      return false;
    }

    // Sort both matrices to ensure consistent ordering
    SparseMatrix sorted_this = Sort();
    SparseMatrix sorted_other = other.Sort();

    return sorted_this.row_indices_ == sorted_other.row_indices_ &&
           sorted_this.col_indices_ == sorted_other.col_indices_ &&
           sorted_this.values_ == sorted_other.values_;
  }
  bool operator!=(const SparseMatrix& other) const { return !(*this == other); }

  // Returns a sorted version of the sparse matrix.
  // The elements are sorted by row index first, then by column index.
  SparseMatrix Sort() const {
    SparseMatrix ret(num_rows_, num_cols_);

    // Create temporary vectors for sorting
    std::vector<uint32_t> indices(row_indices_.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on row and column order
    std::sort(indices.begin(), indices.end(), [this](uint32_t a, uint32_t b) {
      if (row_indices_[a] != row_indices_[b]) {
        return row_indices_[a] < row_indices_[b];
      }
      return col_indices_[a] < col_indices_[b];
    });

    // Fill arrays using sorted indices
    ret.row_indices_.resize(indices.size());
    ret.col_indices_.resize(indices.size());
    ret.values_.resize(indices.size());
    for (uint32_t i = 0; i < indices.size(); ++i) {
      ret.row_indices_[i] = row_indices_[indices[i]];
      ret.col_indices_[i] = col_indices_[indices[i]];
      ret.values_[i] = values_[indices[i]];
    }

    return ret;
  }

  // Inserts or updates an element at (row, col) with value.
  // Returns true if the element was updated, false if it was inserted.
  // CHECK fails if row >= num_rows_ or col >= num_cols_.
  bool Insert(uint32_t row, uint32_t col, const T& value) {
    CHECK_LT(row, num_rows_);
    CHECK_LT(col, num_cols_);

    // Check if element already exists
    for (uint32_t i = 0; i < row_indices_.size(); ++i) {
      if (row_indices_[i] == row && col_indices_[i] == col) {
        values_[i] = value;  // Update existing value
        return true;
      }
    }

    // Insert new element
    row_indices_.push_back(row);
    col_indices_.push_back(col);
    values_.push_back(value);
    return false;
  }

  void InsertUnique(uint32_t row, uint32_t col, const T& value) {
    CHECK_LT(row, num_rows_);
    CHECK_LT(col, num_cols_);

    // Insert new element
    row_indices_.push_back(row);
    col_indices_.push_back(col);
    values_.push_back(value);
  }

  // Convert to CSR format
  void ToCSR(std::vector<uint32_t>& row_ptrs,
             std::vector<uint32_t>& col_indices, std::vector<T>& values,
             bool sort = false) const {
    if (sort) {
      SparseMatrix sorted = Sort();
      sorted.ToCSR(row_ptrs, col_indices, values, false);
      return;
    }

    // Initialize row_ptrs with zeros
    row_ptrs.resize(num_rows_ + 1, 0);

    // Count non-zeros per row
    for (uint32_t i = 0; i < row_indices_.size(); ++i) {
      row_ptrs[row_indices_[i] + 1]++;
    }

    // Compute cumulative sum to get row pointers
    for (uint32_t i = 1; i <= num_rows_; ++i) {
      row_ptrs[i] += row_ptrs[i - 1];
    }

    // Allocate space for col_indices and values
    col_indices.resize(values_.size());
    values.resize(values_.size());

    // Fill col_indices and values arrays directly
    std::vector<uint32_t> row_counts(num_rows_, 0);
    for (uint32_t i = 0; i < row_indices_.size(); ++i) {
      uint32_t row = row_indices_[i];
      uint32_t pos = row_ptrs[row] + row_counts[row];
      col_indices[pos] = col_indices_[i];
      values[pos] = values_[i];
      row_counts[row]++;
    }
  }

  // Convert to CSC format
  void ToCSC(std::vector<uint32_t>& col_ptrs,
             std::vector<uint32_t>& row_indices, std::vector<T>& values,
             bool sort = false) const {
    if (sort) {
      SparseMatrix sorted = Sort();
      sorted.ToCSC(col_ptrs, row_indices, values, false);
      return;
    }

    // Initialize col_ptrs with zeros
    col_ptrs.resize(num_cols_ + 1, 0);

    // Count non-zeros per column
    for (uint32_t i = 0; i < col_indices_.size(); ++i) {
      col_ptrs[col_indices_[i] + 1]++;
    }

    // Compute cumulative sum to get column pointers
    for (uint32_t i = 1; i <= num_cols_; ++i) {
      col_ptrs[i] += col_ptrs[i - 1];
    }

    // Allocate space for row_indices and values
    row_indices.resize(values_.size());
    values.resize(values_.size());

    // Fill row_indices and values arrays directly
    std::vector<uint32_t> col_counts(num_cols_, 0);
    for (uint32_t i = 0; i < col_indices_.size(); ++i) {
      uint32_t col = col_indices_[i];
      uint32_t pos = col_ptrs[col] + col_counts[col];
      row_indices[pos] = row_indices_[i];
      values[pos] = values_[i];
      col_counts[col]++;
    }
  }

  // Convert to COO format
  void ToCOO(std::vector<std::tuple<uint32_t, uint32_t>>& coordinates,
             std::vector<T>& values, bool sort = false) const {
    if (sort) {
      SparseMatrix sorted = Sort();
      sorted.ToCOO(coordinates, values, false);
      return;
    }

    // Simply copy the internal arrays
    coordinates.resize(values_.size());
    for (uint32_t i = 0; i < values_.size(); ++i) {
      coordinates[i] = {row_indices_[i], col_indices_[i]};
    }
    values = values_;
  }

  // Converts the sparse matrix to CSR format and serializes it to a buffer.
  absl::StatusOr<std::vector<uint8_t>> ToCSRBuffer(bool sort = false) const {
    base::AutoReset<bool> auto_reset(
        &base::Serde<std::vector<uint32_t>>::s_ignore_size, true);
    base::AutoReset<bool> auto_reset2(
        &base::Serde<std::vector<T>>::s_ignore_size, true);

    std::vector<uint32_t> row_ptrs, col_indices;
    std::vector<T> values;
    ToCSR(row_ptrs, col_indices, values, sort);
    base::Uint8VectorBuffer buffer;
    TF_RETURN_IF_ERROR(
        buffer.Grow(base::EstimateSize(row_ptrs, col_indices, values)));
    TF_RETURN_IF_ERROR(buffer.WriteMany(row_ptrs, col_indices, values));
    return std::move(buffer).TakeOwnedBuffer();
  }

  // Converts the sparse matrix to CSC format and serializes it to a buffer.
  absl::StatusOr<std::vector<uint8_t>> ToCSCBuffer(bool sort = false) const {
    base::AutoReset<bool> auto_reset(
        &base::Serde<std::vector<uint32_t>>::s_ignore_size, true);
    base::AutoReset<bool> auto_reset2(
        &base::Serde<std::vector<T>>::s_ignore_size, true);

    std::vector<uint32_t> col_ptrs, row_indices;
    std::vector<T> values;
    ToCSC(col_ptrs, row_indices, values, sort);
    base::Uint8VectorBuffer buffer;
    TF_RETURN_IF_ERROR(
        buffer.Grow(base::EstimateSize(col_ptrs, row_indices, values)));
    TF_RETURN_IF_ERROR(buffer.WriteMany(col_ptrs, row_indices, values));
    return std::move(buffer).TakeOwnedBuffer();
  }

  // Converts the sparse matrix to COO format and serializes it to a buffer.
  absl::StatusOr<std::vector<uint8_t>> ToCOOBuffer(bool sort = false) const {
    base::AutoReset<bool> auto_reset(
        &base::Serde<
            std::vector<std::tuple<uint32_t, uint32_t>>>::s_ignore_size,
        true);
    base::AutoReset<bool> auto_reset2(
        &base::Serde<std::vector<T>>::s_ignore_size, true);

    std::vector<std::tuple<uint32_t, uint32_t>> coordinates;
    std::vector<T> values;
    ToCOO(coordinates, values, sort);
    base::Uint8VectorBuffer buffer;
    TF_RETURN_IF_ERROR(buffer.Grow(base::EstimateSize(coordinates, values)));
    TF_RETURN_IF_ERROR(buffer.WriteMany(coordinates, values));
    return std::move(buffer).TakeOwnedBuffer();
  }

  // Deserializes a sparse matrix from a buffer in CSR format.
  static absl::StatusOr<SparseMatrix> FromCSRBuffer(
      const std::vector<uint8_t>& buffer, uint32_t num_rows, uint32_t num_cols,
      size_t num_non_zeros) {
    base::AutoReset<bool> auto_reset(
        &base::Serde<std::vector<uint32_t>>::s_ignore_size, true);
    base::AutoReset<bool> auto_reset2(
        &base::Serde<std::vector<T>>::s_ignore_size, true);

    base::Uint8VectorBuffer buf(buffer);
    std::vector<uint32_t> row_ptrs, col_indices;
    std::vector<T> values;
    row_ptrs.resize(num_rows + 1);
    col_indices.resize(num_non_zeros);
    values.resize(num_non_zeros);
    TF_RETURN_IF_ERROR(buf.ReadMany(&row_ptrs, &col_indices, &values));

    SparseMatrix matrix(num_rows, num_cols);
    for (uint32_t i = 0; i < num_rows; ++i) {
      for (uint32_t j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
        matrix.Insert(i, col_indices[j], values[j]);
      }
    }
    return std::move(matrix);
  }

  // Deserializes a sparse matrix from a buffer in CSC format.
  static absl::StatusOr<SparseMatrix> FromCSCBuffer(
      const std::vector<uint8_t>& buffer, uint32_t num_rows, uint32_t num_cols,
      size_t num_non_zeros) {
    base::AutoReset<bool> auto_reset(
        &base::Serde<std::vector<uint32_t>>::s_ignore_size, true);
    base::AutoReset<bool> auto_reset2(
        &base::Serde<std::vector<T>>::s_ignore_size, true);

    base::Uint8VectorBuffer buf(buffer);
    std::vector<uint32_t> col_ptrs, row_indices;
    std::vector<T> values;
    col_ptrs.resize(num_cols + 1);
    row_indices.resize(num_non_zeros);
    values.resize(num_non_zeros);
    TF_RETURN_IF_ERROR(buf.ReadMany(&col_ptrs, &row_indices, &values));

    SparseMatrix matrix(num_rows, num_cols);
    for (uint32_t j = 0; j < num_cols; ++j) {
      for (uint32_t i = col_ptrs[j]; i < col_ptrs[j + 1]; ++i) {
        matrix.Insert(row_indices[i], j, values[i]);
      }
    }
    return std::move(matrix);
  }

  // Deserializes a sparse matrix from a buffer in COO format.
  static absl::StatusOr<SparseMatrix> FromCOOBuffer(
      const std::vector<uint8_t>& buffer, uint32_t num_rows, uint32_t num_cols,
      size_t num_non_zeros) {
    base::AutoReset<bool> auto_reset(
        &base::Serde<
            std::vector<std::tuple<uint32_t, uint32_t>>>::s_ignore_size,
        true);
    base::AutoReset<bool> auto_reset2(
        &base::Serde<std::vector<T>>::s_ignore_size, true);

    base::Uint8VectorBuffer buf(buffer);
    std::vector<std::tuple<uint32_t, uint32_t>> coordinates;
    std::vector<T> values;
    coordinates.resize(num_non_zeros);
    values.resize(num_non_zeros);
    TF_RETURN_IF_ERROR(buf.ReadMany(&coordinates, &values));

    SparseMatrix matrix(num_rows, num_cols);
    for (uint32_t i = 0; i < num_non_zeros; ++i) {
      matrix.Insert(std::get<0>(coordinates[i]), std::get<1>(coordinates[i]),
                    values[i]);
    }
    return std::move(matrix);
  }

  constexpr static int64_t EstimateCSRSize(uint32_t num_rows, uint32_t num_cols,
                                           uint32_t num_nonzeros) {
    return (num_rows + 1) * sizeof(uint32_t) + num_nonzeros * sizeof(uint32_t) +
           num_nonzeros * sizeof(T);
  }

  constexpr static int64_t EstimateCSCSize(uint32_t num_rows, uint32_t num_cols,
                                           uint32_t num_nonzeros) {
    return (num_cols + 1) * sizeof(uint32_t) + num_nonzeros * sizeof(uint32_t) +
           num_nonzeros * sizeof(T);
  }

  constexpr static int64_t EstimateCOOSize(uint32_t num_rows, uint32_t num_cols,
                                           uint32_t num_nonzeros) {
    return num_nonzeros * sizeof(uint32_t) * 2 + num_nonzeros * sizeof(T);
  }

  std::string ToString() const {
    std::stringstream ss;
    ss << "SparseMatrix(" << num_rows_ << "x" << num_cols_
       << ", nnz=" << values_.size() << ")\n";

    for (uint32_t i = 0; i < values_.size(); ++i) {
      ss << "(" << row_indices_[i] << "," << col_indices_[i]
         << ") = " << values_[i] << "\n";
    }
    return ss.str();
  }

 private:
  // COO (Coordinate) format storage
  std::vector<uint32_t> row_indices_;  // Row indices of non-zero elements
  std::vector<uint32_t> col_indices_;  // Column indices of non-zero elements
  std::vector<T> values_;              // Values of non-zero elements
  uint32_t num_rows_ = 0;              // Number of rows in matrix
  uint32_t num_cols_ = 0;              // Number of columns in matrix
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const SparseMatrix<T>& matrix) {
  return os << matrix.ToString();
}

}  // namespace zkx::math

#endif  // ZKX_MATH_BASE_SPARSE_MATRIX_H_
