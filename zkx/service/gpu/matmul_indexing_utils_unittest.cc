/* Copyright 2022 The OpenXLA Authors.

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

#include "gtest/gtest.h"

#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/parser/hlo_parser.h"

namespace zkx::gpu {
namespace {

using ::testing::ElementsAre;
using ::tsl::testing::IsOkAndHolds;

TEST(MatMulIndexingUtilsTest, GetNonContractingDims) {
  TF_ASSERT_OK_AND_ASSIGN(Shape shape, ParseShape("u32[1,2,3,4,5,6]"))
  EXPECT_THAT(GetNonContractingDims(shape, /*batch_dims=*/{4},
                                    /*contracting_dims=*/{1, 5}),
              IsOkAndHolds(ElementsAre(0, 2, 3)));
}

TEST(MatMulIndexingUtilsTest, BatchDimensionsForOperand) {
  const std::string hlo_text = R"(
    HloModule m

    ENTRY main {
      %lhs = u32[16,32,128] parameter(0)
      %rhs = u32[16,128,64] parameter(1)
      ROOT %r = u32[16,32,64] dot(%lhs, %rhs),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_text));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  const auto& lhs_batch_dims = BatchDimensionsForOperand(*dot, 0);
  EXPECT_THAT(
      std::vector<int64_t>(lhs_batch_dims.begin(), lhs_batch_dims.end()),
      ElementsAre(0));

  const auto& rhs_batch_dims = BatchDimensionsForOperand(*dot, 1);
  EXPECT_THAT(
      std::vector<int64_t>(rhs_batch_dims.begin(), rhs_batch_dims.end()),
      ElementsAre(0));
}

TEST(MatMulIndexingUtilsTest, ContractingDimensionIndex) {
  const std::string hlo_text = R"(
    HloModule m

    ENTRY main {
      %lhs = u32[16,32,128] parameter(0)
      %rhs = u32[16,128,64] parameter(1)
      ROOT %r = u32[16,32,64] dot(%lhs, %rhs),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_text));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  TF_ASSERT_OK_AND_ASSIGN(int64_t lhs_contracting,
                          ContractingDimensionIndex(*dot, 0));
  EXPECT_EQ(lhs_contracting, 2);

  TF_ASSERT_OK_AND_ASSIGN(int64_t rhs_contracting,
                          ContractingDimensionIndex(*dot, 1));
  EXPECT_EQ(rhs_contracting, 1);
}

TEST(MatMulIndexingUtilsTest, NonContractingDimensionIndex) {
  const std::string hlo_text = R"(
    HloModule m

    ENTRY main {
      %lhs = u32[16,32,128] parameter(0)
      %rhs = u32[16,128,64] parameter(1)
      ROOT %r = u32[16,32,64] dot(%lhs, %rhs),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_text));
  const HloInstruction* dot = module->entry_computation()->root_instruction();

  TF_ASSERT_OK_AND_ASSIGN(int64_t lhs_non_contracting,
                          NonContractingDimensionIndex(*dot, 0));
  EXPECT_EQ(lhs_non_contracting, 1);

  TF_ASSERT_OK_AND_ASSIGN(int64_t rhs_non_contracting,
                          NonContractingDimensionIndex(*dot, 1));
  EXPECT_EQ(rhs_non_contracting, 2);
}

}  // namespace
}  // namespace zkx::gpu
