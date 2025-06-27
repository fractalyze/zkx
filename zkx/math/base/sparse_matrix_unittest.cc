#include "zkx/math/base/sparse_matrix.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"

namespace zkx::math {
namespace {

SparseMatrix<int> CreateTestSparseMatrix() {
  SparseMatrix<int> matrix(3, 4);

  matrix.Insert(2, 1, 5);
  matrix.Insert(0, 3, 2);
  matrix.Insert(1, 0, 7);
  matrix.Insert(0, 1, 3);
  matrix.Insert(2, 0, 1);

  return matrix;
}
}  // namespace

TEST(SparseMatrixTest, BasicOperations) {
  // Create a 3x4 sparse matrix
  SparseMatrix<int> matrix(3, 4);

  // Test initial state
  EXPECT_EQ(matrix.num_rows(), 3);
  EXPECT_EQ(matrix.num_cols(), 4);
  EXPECT_EQ(matrix.NumNonZeros(), 0);

  // Insert some elements
  EXPECT_FALSE(matrix.Insert(0, 1, 5));
  EXPECT_FALSE(matrix.Insert(1, 2, 7));
  EXPECT_FALSE(matrix.Insert(2, 0, 3));
  // Update existing element
  EXPECT_TRUE(matrix.Insert(0, 1, 6));

  // Test number of non-zeros
  EXPECT_EQ(matrix.NumNonZeros(), 3);

  // Test element access
  EXPECT_EQ(*matrix(0, 1), 6);  // Updated value
  EXPECT_EQ(*matrix(1, 2), 7);
  EXPECT_EQ(*matrix(2, 0), 3);
  EXPECT_EQ(matrix(0, 0), nullptr);  // Non-existent element
  EXPECT_EQ(matrix(1, 1), nullptr);  // Non-existent element
}

TEST(SparseMatrixTest, Sort) {
  SparseMatrix<int> matrix = CreateTestSparseMatrix();
  SparseMatrix<int> sorted = matrix.Sort();

  EXPECT_EQ(sorted.num_rows(), matrix.num_rows());
  EXPECT_EQ(sorted.num_cols(), matrix.num_cols());
  EXPECT_EQ(sorted.NumNonZeros(), matrix.NumNonZeros());

  std::vector<std::tuple<uint32_t, uint32_t>> coordinates;
  std::vector<int> values;
  sorted.ToCOO(coordinates, values, false);

  EXPECT_THAT(coordinates,
              testing::ElementsAre(std::make_tuple(0, 1), std::make_tuple(0, 3),
                                   std::make_tuple(1, 0), std::make_tuple(2, 0),
                                   std::make_tuple(2, 1)));
  EXPECT_THAT(values, testing::ElementsAre(3, 2, 7, 1, 5));
}

TEST(SparseMatrixTest, FormatConversion) {
  SparseMatrix<int> matrix = CreateTestSparseMatrix();

  // Test COO format without sorting
  {
    std::vector<std::tuple<uint32_t, uint32_t>> coordinates;
    std::vector<int> values;
    matrix.ToCOO(coordinates, values, false);

    EXPECT_THAT(coordinates, testing::ElementsAre(
                                 std::make_tuple(2, 1), std::make_tuple(0, 3),
                                 std::make_tuple(1, 0), std::make_tuple(0, 1),
                                 std::make_tuple(2, 0)));
    EXPECT_THAT(values, testing::ElementsAre(5, 2, 7, 3, 1));
  }

  // Test COO format with sorting
  {
    std::vector<std::tuple<uint32_t, uint32_t>> coordinates;
    std::vector<int> values;
    matrix.ToCOO(coordinates, values, true);

    EXPECT_THAT(coordinates, testing::ElementsAre(
                                 std::make_tuple(0, 1), std::make_tuple(0, 3),
                                 std::make_tuple(1, 0), std::make_tuple(2, 0),
                                 std::make_tuple(2, 1)));
    EXPECT_THAT(values, testing::ElementsAre(3, 2, 7, 1, 5));
  }

  // Test CSR format without sorting
  {
    std::vector<uint32_t> row_ptrs, col_indices;
    std::vector<int> values;
    matrix.ToCSR(row_ptrs, col_indices, values, false);

    EXPECT_THAT(row_ptrs, testing::ElementsAre(0, 2, 3, 5));
    EXPECT_THAT(col_indices, testing::ElementsAre(3, 1, 0, 1, 0));
    EXPECT_THAT(values, testing::ElementsAre(2, 3, 7, 5, 1));
  }

  // Test CSR format with sorting
  {
    std::vector<uint32_t> row_ptrs, col_indices;
    std::vector<int> values;
    matrix.ToCSR(row_ptrs, col_indices, values, true);

    EXPECT_THAT(row_ptrs, testing::ElementsAre(0, 2, 3, 5));
    EXPECT_THAT(col_indices, testing::ElementsAre(1, 3, 0, 0, 1));
    EXPECT_THAT(values, testing::ElementsAre(3, 2, 7, 1, 5));
  }

  // Test CSC format without sorting
  {
    std::vector<uint32_t> col_ptrs, row_indices;
    std::vector<int> values;
    matrix.ToCSC(col_ptrs, row_indices, values, false);

    EXPECT_THAT(col_ptrs, testing::ElementsAre(0, 2, 4, 4, 5));
    EXPECT_THAT(row_indices, testing::ElementsAre(1, 2, 2, 0, 0));
    EXPECT_THAT(values, testing::ElementsAre(7, 1, 5, 3, 2));
  }

  // Test CSC format with sorting
  {
    std::vector<uint32_t> col_ptrs, row_indices;
    std::vector<int> values;
    matrix.ToCSC(col_ptrs, row_indices, values, true);

    EXPECT_THAT(col_ptrs, testing::ElementsAre(0, 2, 4, 4, 5));
    EXPECT_THAT(row_indices, testing::ElementsAre(1, 2, 0, 2, 0));
    EXPECT_THAT(values, testing::ElementsAre(7, 1, 3, 5, 2));
  }
}

TEST(SparseMatrixTest, OutOfBounds) {
  SparseMatrix<int> matrix(2, 2);

  // Test out of bounds access
  EXPECT_DEATH(matrix(2, 0), "");            // Row out of bounds
  EXPECT_DEATH(matrix(0, 2), "");            // Column out of bounds
  EXPECT_DEATH(matrix.Insert(2, 0, 1), "");  // Row out of bounds
  EXPECT_DEATH(matrix.Insert(0, 2, 1), "");  // Column out of bounds
}

TEST(SparseMatrixTest, ToFromBuffers) {
  auto matrix = CreateTestSparseMatrix();
  uint32_t num_rows = matrix.num_rows();
  uint32_t num_cols = matrix.num_cols();
  uint32_t num_nonzeros = matrix.NumNonZeros();

  std::vector<uint8_t> buffer;
  SparseMatrix<int> reconstructed;
  TF_ASSERT_OK_AND_ASSIGN(buffer, matrix.ToCSRBuffer());
  TF_ASSERT_OK_AND_ASSIGN(
      reconstructed, SparseMatrix<int>::FromCSRBuffer(buffer, num_rows,
                                                      num_cols, num_nonzeros));
  EXPECT_EQ(reconstructed, matrix);

  TF_ASSERT_OK_AND_ASSIGN(buffer, matrix.ToCSCBuffer());
  TF_ASSERT_OK_AND_ASSIGN(
      reconstructed, SparseMatrix<int>::FromCSCBuffer(buffer, num_rows,
                                                      num_cols, num_nonzeros));
  EXPECT_EQ(reconstructed, matrix);

  TF_ASSERT_OK_AND_ASSIGN(buffer, matrix.ToCOOBuffer());
  TF_ASSERT_OK_AND_ASSIGN(
      reconstructed, SparseMatrix<int>::FromCOOBuffer(buffer, num_rows,
                                                      num_cols, num_nonzeros));
  EXPECT_EQ(reconstructed, matrix);
}

}  // namespace zkx::math
