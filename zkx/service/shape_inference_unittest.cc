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

#include <array>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zkx/hlo/parser/hlo_parser.h"

namespace zkx {

using ::testing::ContainsRegex;
using ::testing::HasSubstr;

constexpr std::string_view kBroadcastDimensionMismatchErrorMessage =
    "Broadcast dimension 0 mismatch";
constexpr std::string_view kIncompatibleBinaryOpShapeErrorMessage =
    "Binary op with incompatible shapes";
std::array<const int64_t, 1> zero_array = {0};

class ShapeInferenceTest : public ::testing::Test {
 protected:
  // Some handy scalar shapes.
  const Shape s32_ = ShapeUtil::MakeShape(S32, {});
  const Shape u16_ = ShapeUtil::MakeShape(U16, {});
  const Shape u32_ = ShapeUtil::MakeShape(U32, {});
  const Shape u64_ = ShapeUtil::MakeShape(U64, {});
  const Shape pred_ = ShapeUtil::MakeShape(PRED, {});

  // Some handy vector and matrix shapes of U32 type.
  // Suffix: vector_length_, matrix_rows_cols_
  const Shape vector_32_ = ShapeUtil::MakeShape(U32, {32});
  const Shape vector_64_ = ShapeUtil::MakeShape(U32, {64});
  const Shape matrix_32_48_ = ShapeUtil::MakeShape(U32, {32, 48});
  const Shape matrix_32_64_ = ShapeUtil::MakeShape(U32, {32, 64});
  const Shape matrix_64_48_ = ShapeUtil::MakeShape(U32, {64, 48});

  // Some handy S32 arrays.
  const Shape s32matrix_64_64_ = ShapeUtil::MakeShape(S32, {64, 64});
};

struct BinaryOpTestCase {
  std::string lhs;
  std::string rhs;
  absl::Span<const int64_t> broadcast_dimensions;
  std::string expected;
  std::optional<std::string_view> error_message;
};

// Subclass for testing unbounded dynamic logical ops
class UnboundedLogicalOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic binary ops
class UnboundedBinaryOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic compare op
class UnboundedCompareOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic complex op
class UnboundedComplexOpShapeInferenceTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

// Subclass for testing unbounded dynamic concatenate op
class UnboundedConcatenateOpShapeInferenceTest
    : public ::testing::TestWithParam<std::vector<std::string>> {};

struct UnaryOpTestCase {
  std::string operand;
  std::string expected;
  HloOpcode opcode;
};

// Subclass for testing unbounded dynamic unary ops
class UnboundedUnaryOpShapeInferenceTest
    : public ::testing::TestWithParam<UnaryOpTestCase> {};

// Subclass for testing unbounded dynamic clamp op
class UnboundedClampOpShapeInferenceTest
    : public ::testing::TestWithParam<std::vector<std::string>> {};

// Subclass for testing unbounded dynamic select op
class UnboundedSelectOpShapeInferenceTest
    : public ::testing::TestWithParam<std::vector<std::string>> {};

TEST_F(ShapeInferenceTest, UnaryNegateMatrix) {
  const Shape matrix_shape = ShapeUtil::MakeShape(U32, {128, 64});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferUnaryOpShape(HloOpcode::kNegate, matrix_shape);
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(matrix_shape, *inferred_shape));
}

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferTernaryOpShape
// clang-format on

TEST_F(ShapeInferenceTest, VariadicOpTuplify) {
  const absl::StatusOr<Shape> result =
      ShapeInference::InferVariadicOpShape(HloOpcode::kTuple, {&s32_, &u32_});
  ASSERT_TRUE(result.status().ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(*result, ShapeUtil::MakeTupleShape({s32_, u32_})));
}

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferSelectAndScatterShape
// clang-format on

TEST_F(ShapeInferenceTest, AllGatherStart) {
  const Shape operand = ShapeUtil::MakeShape(U32, {1, 8, 4});
  const Shape expected_shape = ShapeUtil::MakeTupleShape(
      {operand, ShapeUtil::MakeShape(U32, {8, 8, 4})});

  const absl::StatusOr<Shape> inferred_ag_shape =
      ShapeInference::InferAllGatherStartShape(
          {&operand}, /*all_gather_dimension=*/0, /*shard_count=*/8);
  EXPECT_TRUE(inferred_ag_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, AllGatherStartMultiOperand) {
  const Shape operand0 = ShapeUtil::MakeShape(U32, {1, 8, 4});
  const Shape operand1 = ShapeUtil::MakeShape(U16, {1, 5});
  const Shape expected_output0_shape = ShapeUtil::MakeShape(U32, {8, 8, 4});
  const Shape expected_output1_shape = ShapeUtil::MakeShape(U16, {8, 5});
  const Shape expected_shape = ShapeUtil::MakeTupleShape(
      {/* tuple of all input shapes*/
       ShapeUtil::MakeTupleShape({operand0, operand1}),
       /* tuple of all output shapes*/
       ShapeUtil::MakeTupleShape(
           {expected_output0_shape, expected_output1_shape})});

  const absl::StatusOr<Shape> inferred_ag_shape =
      ShapeInference::InferAllGatherStartShape({&operand0, &operand1},
                                               /*all_gather_dimension=*/0,
                                               /*shard_count=*/8);
  EXPECT_TRUE(inferred_ag_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, AllGatherDone) {
  const Shape input_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(U32, {1, 8, 4}),
                                 ShapeUtil::MakeShape(U32, {8, 8, 4})});
  const Shape expected_shape = ShapeUtil::MakeShape(U32, {8, 8, 4});

  const absl::StatusOr<Shape> inferred_ag_done_shape =
      ShapeInference::InferAllGatherDoneShape(input_shape);
  EXPECT_TRUE(inferred_ag_done_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_done_shape, expected_shape));
}

TEST_F(ShapeInferenceTest, AllGatherDoneMultiOperand) {
  const Shape operand0 = ShapeUtil::MakeShape(U32, {1, 8, 4});
  const Shape operand1 = ShapeUtil::MakeShape(U16, {1, 5});
  const Shape expected_output0_shape = ShapeUtil::MakeShape(U32, {8, 8, 4});
  const Shape expected_output1_shape = ShapeUtil::MakeShape(U16, {8, 5});
  const Shape input_shape = ShapeUtil::MakeTupleShape(
      {/* tuple of all input shapes*/
       ShapeUtil::MakeTupleShape({operand0, operand1}),
       /* tuple of all output shapes*/
       ShapeUtil::MakeTupleShape(
           {expected_output0_shape, expected_output1_shape})});

  const Shape expected_shape = ShapeUtil::MakeTupleShape(
      {expected_output0_shape, expected_output1_shape});

  const absl::StatusOr<Shape> inferred_ag_done_shape =
      ShapeInference::InferAllGatherDoneShape(input_shape);
  EXPECT_TRUE(inferred_ag_done_shape.ok());
  EXPECT_TRUE(ShapeUtil::Equal(*inferred_ag_done_shape, expected_shape));
}

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferMapShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferReduceShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferSliceShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferGetTupleElementShape
// clang-format on

TEST_F(ShapeInferenceTest, InferPowShape) {
  const Shape ten_u32s = ShapeUtil::MakeShape(U32, {10});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kPower, ten_u32s, u32_, {});
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(ten_u32s, *inferred_shape));
}

TEST_F(ShapeInferenceTest, InferCompareShape) {
  const Shape ten_u32s = ShapeUtil::MakeShape(U32, {10});
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kCompare, ten_u32s, u32_,
                                         {});
  ASSERT_TRUE(inferred_shape.status().ok());
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(PRED, {10}), *inferred_shape));
}

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferReshapeShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferBroadcastShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferDotOpShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferSetDimensionSizeShape
// clang-format on

TEST_F(ShapeInferenceTest, BinOpBroadcastMatrixVector) {
  // Test variations of broadcasting a vector for a binary add with a
  // matrix.
  const Shape mat = ShapeUtil::MakeShape(U32, {16, 8});
  const Shape vec8 = ShapeUtil::MakeShape(U32, {8});
  const Shape vec16 = ShapeUtil::MakeShape(U32, {16});

  absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec8, {1});
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, mat));

  absl::StatusOr<Shape> inferred_shape_mismatch =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec8, {0});
  ASSERT_FALSE(inferred_shape_mismatch.ok());

  inferred_shape_match =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec16, {0});
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, mat));

  inferred_shape_mismatch =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, mat, vec16, {1});
  ASSERT_FALSE(inferred_shape_mismatch.ok());
}

TEST_F(ShapeInferenceTest, BinOpBroadcastCubeMatrix) {
  // Test variations of broadcasting a matrix for a binary add with a cube.
  const Shape cube = ShapeUtil::MakeShape(U32, {16, 8, 4});
  const Shape matrix8_4 = ShapeUtil::MakeShape(U32, {8, 4});
  const Shape matrix16_4 = ShapeUtil::MakeShape(U32, {16, 4});
  const Shape matrix16_8 = ShapeUtil::MakeShape(U32, {16, 8});

  absl::StatusOr<Shape> inferred_shape_match =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, cube, matrix8_4,
                                         {1, 2});
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, cube));

  inferred_shape_match = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, cube, matrix16_4, {0, 2});
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, cube));

  inferred_shape_match = ShapeInference::InferBinaryOpShape(
      HloOpcode::kAdd, cube, matrix16_8, {0, 1});
  ASSERT_TRUE(inferred_shape_match.status().ok());
  ASSERT_TRUE(ShapeUtil::Equal(*inferred_shape_match, cube));
}

TEST_F(ShapeInferenceTest, BinOpBroadcastBadDimension) {
  // Test various errors with the broadcast argument.
  const Shape tensor = ShapeUtil::MakeShape(U32, {16, 8, 4});
  const Shape tensor8_8_8 = ShapeUtil::MakeShape(U32, {8, 8, 8});
  const Shape vec8 = ShapeUtil::MakeShape(U32, {8});
  const Shape matrix8_4 = ShapeUtil::MakeShape(U32, {8, 4});
  const Shape matrix8_8 = ShapeUtil::MakeShape(U32, {8, 8});

  // "magical" broadcast rejected
  const absl::StatusOr<Shape> inferred_shape_error1 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {});
  ASSERT_FALSE(inferred_shape_error1.ok());
  ASSERT_THAT(inferred_shape_error1.status().message(),
              HasSubstr("Shapes must be equal rank"));

  // broadcast_dimension out of bounds for tensor's rank
  const absl::StatusOr<Shape> inferred_shape_error2 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {3});
  ASSERT_FALSE(inferred_shape_error2.ok());
  ASSERT_THAT(inferred_shape_error2.status().message(),
              ContainsRegex("Broadcast dimension number .* too large"));

  // broadcast_dimension doesn't match corresponding dimension
  const absl::StatusOr<Shape> inferred_shape_error3 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, vec8, {0});
  ASSERT_FALSE(inferred_shape_error3.ok());
  ASSERT_THAT(inferred_shape_error3.status().message(),
              HasSubstr("Broadcast dimension 0 mismatch"));

  // broadcast_dimensions list too long
  const absl::StatusOr<Shape> inferred_shape_error4 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, matrix8_4,
                                         {0, 1, 2});
  ASSERT_FALSE(inferred_shape_error4.ok());
  ASSERT_THAT(inferred_shape_error4.status().message(),
              HasSubstr("broadcast_dimensions has to match"));

  // there's a dimension above the rank of the tensor
  const absl::StatusOr<Shape> inferred_shape_error5 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, matrix8_4,
                                         {3, 0});
  ASSERT_FALSE(inferred_shape_error5.ok());
  ASSERT_THAT(inferred_shape_error5.status().message(),
              ContainsRegex("dimension number .* too large"));

  // broadcasting dimensions don't match in this order
  const absl::StatusOr<Shape> inferred_shape_error6 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor, matrix8_4,
                                         {2, 1});
  ASSERT_FALSE(inferred_shape_error6.ok());
  ASSERT_THAT(inferred_shape_error6.status().message(),
              HasSubstr("dimension 0 mismatch"));

  // The following two tests make sure that broadcasting dimensions are listed
  // in a proper (strictly increasing) order, even if the lower-rank array
  // matches the higher-rank array in many different ways.
  const absl::StatusOr<Shape> inferred_shape_error7 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor8_8_8,
                                         matrix8_8, {0, 0});
  ASSERT_FALSE(inferred_shape_error7.ok());
  ASSERT_THAT(inferred_shape_error7.status().message(),
              HasSubstr("dimensions order is wrong"));

  const absl::StatusOr<Shape> inferred_shape_error8 =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, tensor8_8_8,
                                         matrix8_8, {1, 0});
  ASSERT_FALSE(inferred_shape_error8.ok());
  ASSERT_THAT(inferred_shape_error8.status().message(),
              HasSubstr("dimensions order is wrong"));
}

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferWhileShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferConcatOpShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferPadShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferReverseShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferCallShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferTransposeShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferConditionalShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferSliceShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: HloOpcode::kSort
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferTopKShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferGatherShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferScatterShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: HloParser::ParseShape
// clang-format on

TEST_P(UnboundedUnaryOpShapeInferenceTest, UnboundedUnaryOps) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape(GetParam().operand));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape(GetParam().expected));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred,
      ShapeInference::InferUnaryOpShape(GetParam().opcode, operand));
  EXPECT_TRUE(ShapeUtil::Equal(inferred, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedAdd) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_F(ShapeInferenceTest, UnboundedAllGather) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferAllGatherShape(
          {&operand}, /*all_gather_dimension=*/0, /*shard_count=*/2));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedAllReduce) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferAllReduceShape({&operand}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedAllToAll) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferAllToAllShape(/*shape=*/operand,
                                         /*split_dimension=*/0,
                                         /*concat_dimension=*/0,
                                         /*split_count=*/3));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedAllToAllTupleUnsupported) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("(u32[?, 10], u32[?, 10])"));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferAllToAllTupleShape(
          /*operand_shapes=*/{&operand, &operand});
  EXPECT_THAT(
      inferred_shape.status().message(),
      HasSubstr("AllToAllTuple does not support unbounded dynamic shapes"));
}

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: HloOpcode::kAnd
// clang-format on
// TEST_P(UnboundedLogicalOpShapeInferenceTest, UnboundedAnd) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ShapeInference::InferBitcastConvertShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedBitcastConvert) {

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferBroadcastShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferCallShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferTernaryOpShape
// clang-format on

TEST_F(ShapeInferenceTest, UnboundedCollectiveBroadcast) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferCollectiveBroadcastShape(
                              /*operand_shapes=*/{&operand}));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/{&operand}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CollectivePermuteStart) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("(u32[8, 8], u32[8, 8])"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteStartShape(
          /*operand_shapes=*/{&operand}, {}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CombinedCollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_0, ParseShape("u32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_1, ParseShape("u32[16, 16]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("(u32[8, 8], u32[16, 16])"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/{&operand_0, &operand_1}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CombinedCollectivePermuteStart) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_0, ParseShape("u32[8, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand_1, ParseShape("u32[16, 16]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape expected,
      ParseShape("((u32[8, 8], u32[16, 16]), (u32[8, 8], u32[16, 16]))"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferCollectivePermuteStartShape(
                              /*operand_shapes=*/{&operand_0, &operand_1}, {},
                              /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, CombinedInplaceCollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand,
                          ParseShape("(u32[2,3], u32[2,3], u32[], u32[])"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[2,3]"));
  std::vector<const Shape*> operand_shapes;
  absl::c_transform(operand.tuple_shapes(), std::back_inserter(operand_shapes),
                    [](const Shape& shape) { return &shape; });
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/operand_shapes, /*inplace=*/true));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_F(ShapeInferenceTest, UnboundedCollectivePermute) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape inferred_shape,
      ShapeInference::InferCollectivePermuteShape(
          /*operand_shapes=*/{&operand}, /*inplace=*/false));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

TEST_P(UnboundedCompareOpShapeInferenceTest, UnboundedCompare) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kCompare, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferConcatOpShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferConvertShape
// clang-format on
TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedDiv) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kDivide, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferDotOpShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferDynamicSliceShape
// clang-format on

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferDynamicUpdateSliceShape
// clang-format on

// clang-format off
// TODO(chokobole): Uncomment comment. Dependency: ShapeInference::InferGatherShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedGather) {

// clang-format off
// TODO(chokobole): Uncomment comment. Dependency: ShapeInference::InferGetTupleElementShape
// clang-format on
// TEST(XlaBuilderTest, UnboundedGetTupleElement) {

// clang-format off
// TODO(chokobole): Uncomment comment. Dependency: ShapeInference::InferMapShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedMap) {

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedMax) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kMaximum, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedMin) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_status =
      ShapeInference::InferBinaryOpShape(HloOpcode::kMinimum, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_status.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_status, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_status)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_status.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedMul) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kMultiply, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

// clang-format off
// TODO(chokobole): Uncomment comment. Dependency: HloOpcode::kOr
// clang-format on
// TEST_P(UnboundedLogicalOpShapeInferenceTest, UnboundedOr) {

// clang-format off
// TODO(chokobole): Uncomment comment. Dependency: ShapeInference::InferPadShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedPad) {

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedPow) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kPower, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferReduceShape
// clang-format on

TEST_F(ShapeInferenceTest, UnboundedReduceScatter) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape inferred_shape,
                          ShapeInference::InferReduceScatterShape(
                              /*operand_shapes=*/{&operand},
                              /*scatter_dimension=*/0, /*shard_count=*/2));
  EXPECT_TRUE(ShapeUtil::Equal(inferred_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(inferred_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: HloOpcode::kRemainder
// clang-format on
// TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedRemainder) {

// clang-format off
// TODO(chokobole): Add tests. Dependency: ShapeInference::InferReshapeShape
// clang-format on

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ShapeInference::InferReverseShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedReverse) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ShapeInference::InferScatterShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedScatter) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ShapeInference::InferTernaryOpShape
// clang-format on
// TEST_P(UnboundedSelectOpShapeInferenceTest, UnboundedSelect) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ShapeInference::InferTernaryOpShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedSelectWithTupleUnsupported) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ShapeInference::InferSelectAndScatterShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedSelectAndScatter) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: HloOpcode::kShiftLeft
// clang-format on
// TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedShiftLeft) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: HloOpcode::kShiftRightArithmetic
// clang-format on
// TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedShiftRightArithmetic) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: HloOpcode::kShiftRightLogical
// clang-format on
// TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedShiftRightLogical) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ShapeInference::InferSliceShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedSlice) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: HloOpcode::kSort
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedSort) {

TEST_P(UnboundedBinaryOpShapeInferenceTest, UnboundedSub) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  const absl::StatusOr<Shape> inferred_shape =
      ShapeInference::InferBinaryOpShape(HloOpcode::kSubtract, lhs, rhs,
                                         GetParam().broadcast_dimensions);
  if (inferred_shape.ok()) {
    TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                            ParseShape(GetParam().expected));
    EXPECT_TRUE(ShapeUtil::Equal(*inferred_shape, expected))
        << "inferred: " << ShapeUtil::HumanString(*inferred_shape)
        << " expected: " << ShapeUtil::HumanString(expected);
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(inferred_shape.status().message(),
                HasSubstr(*GetParam().error_message));
  }
}

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ShapeInference::InferTransposeShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedTranspose) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ShapeInference::InferTransposeShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedTransposeRank1) {

TEST_F(ShapeInferenceTest, UnboundedTuple) {
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("u32[?, 10]"));
  const Shape expected = ShapeUtil::MakeTupleShape({operand});
  TF_ASSERT_OK_AND_ASSIGN(
      const Shape result_shape,
      ShapeInference::InferVariadicOpShape(
          HloOpcode::kTuple, std::vector<const Shape*>({&operand})));
  EXPECT_TRUE(ShapeUtil::Equal(result_shape, expected))
      << "inferred: " << ShapeUtil::HumanString(result_shape)
      << " expected: " << ShapeUtil::HumanString(expected);
}

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ShapeInference::InferWhileShape
// clang-format on
// TEST_F(ShapeInferenceTest, UnboundedWhile) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: HloOpcode::kXor
// clang-format on
// TEST_P(UnboundedLogicalOpShapeInferenceTest, UnboundedXor) {

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism, UnboundedBinaryOpShapeInferenceTest,
                         ::testing::ValuesIn<BinaryOpTestCase>(
                             {// LHS | RHS | bdims | Res
                              // 1   | ?   | []    | ?
                              {"u32[1]", "u32[?]", {}, "u32[?]"},
                              // ?   | 1   | []    | ?
                              {"u32[?]", "u32[1]", {}, "u32[?]"},
                              // 2   | ?   | []    | 2
                              {"u32[2]", "u32[?]", {}, "u32[2]"},
                              // ?   | 2   | []    | 2
                              {"u32[?]", "u32[2]", {}, "u32[2]"},
                              // <=2 | ?   | []    | <=2
                              {"u32[<=2]", "u32[?]", {}, "u32[<=2]"},
                              // ?   | <=2 | []    | <=2
                              {"u32[?]", "u32[<=2]", {}, "u32[<=2]"},
                              // ?   | ?   | []    | ?
                              {"u32[?]", "u32[?]", {}, "u32[?]"},
                              // 1   | ?,3 | [0]   | ?,3
                              {"u32[1]", "u32[?,3]", zero_array, "u32[?,3]"},
                              // 2   | ?,3 | [0]   | err
                              {"u32[2]", "u32[?,3]", zero_array, "",
                               kBroadcastDimensionMismatchErrorMessage},
                              // ?,2 | ?,3 | []    | err
                              {"u32[?,2]",
                               "u32[?,3]",
                               {},
                               "",
                               kIncompatibleBinaryOpShapeErrorMessage}}));

INSTANTIATE_TEST_SUITE_P(UnboundedDynamism,
                         UnboundedCompareOpShapeInferenceTest,
                         ::testing::ValuesIn<BinaryOpTestCase>(
                             {// LHS | RHS | bdims | Res
                              // 1   | ?   | []    | ?
                              {"u32[1]", "u32[?]", {}, "pred[?]"},
                              // ?   | 1   | []    | ?
                              {"u32[?]", "u32[1]", {}, "pred[?]"},
                              // 2   | ?   | []    | 2
                              {"u32[2]", "u32[?]", {}, "pred[2]"},
                              // ?   | 2   | []    | 2
                              {"u32[?]", "u32[2]", {}, "pred[2]"},
                              // <=2 | ?   | []    | <=2
                              {"u32[<=2]", "u32[?]", {}, "pred[<=2]"},
                              // ?   | <=2 | []    | <=2
                              {"u32[?]", "u32[<=2]", {}, "pred[<=2]"},
                              // ?   | ?   | []    | ?
                              {"u32[?]", "u32[?]", {}, "pred[?]"},
                              // 1   | ?,3 | [0]   | ?,3
                              {"u32[1]", "u32[?,3]", zero_array, "pred[?,3]"},
                              // 2   | ?,3 | [0]   | err
                              {"u32[2]", "u32[?,3]", zero_array, "",
                               kBroadcastDimensionMismatchErrorMessage},
                              // ?,2 | ?,3 | []    | err
                              {"u32[?,2]",
                               "u32[?,3]",
                               {},
                               "",
                               kIncompatibleBinaryOpShapeErrorMessage}}));

INSTANTIATE_TEST_SUITE_P(
    UnboundedDynamism, UnboundedUnaryOpShapeInferenceTest,
    ::testing::ValuesIn<UnaryOpTestCase>({
        // TODO(chokobole): Uncomment this. Dependency: HloOpcode::kAbs
        //   {"u32[?]", "u32[?]", HloOpcode::kAbs},
        {"u32[?]", "u32[?]", HloOpcode::kNegate},
        // TODO(chokobole): Uncomment this. Dependency: HloOpcode::kAbs
        //  {"s32[?]", "s32[?]", HloOpcode::kNot},
        // TODO(chokobole): Uncomment this. Dependency:
        // HloOpcode::kPopulationCount
        //  {"u32[?]", "u32[?]", HloOpcode::kPopulationCount},
        // TODO(chokobole): Uncomment this. Dependency: HloOpcode::kSign
        //  {"u32[?]", "u32[?]", HloOpcode::kSign},
    }));

}  // namespace zkx
