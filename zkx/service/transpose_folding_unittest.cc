/* Copyright 2017 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#include "zkx/service/transpose_folding.h"

#include <memory>

#include "absl/status/status_matchers.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/builder/zkx_builder.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/utils/hlo_matchers.h"
#include "zkx/tests/hlo_test_base.h"

namespace zkx {
namespace {

namespace op = testing::opcode_matchers;
using ::absl_testing::IsOkAndHolds;

using TransposeFoldingTest = HloTestBase;

TEST_F(TransposeFoldingTest, FoldDotTranspose) {
  constexpr std::string_view kHloString = R"(
HloModule FoldDotTranspose

ENTRY entry_computation {
  x = s32[2,3]{1,0} parameter(0)
  y = s32[2,3]{1,0} parameter(1)
  transpose = s32[3,2]{1,0} transpose(y), dimensions={1,0}
  ROOT dot = s32[2,2]{1,0} dot(x, transpose), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Parameter(0), op::Parameter(1),
                      /*lhs_contracting_dim=*/1, /*rhs_contracting_dim=*/1));
}

TEST_F(TransposeFoldingTest, DontFoldTransposeOfBatchDimByDefault) {
  constexpr std::string_view kHloString = R"(
HloModule FoldDotTranspose

ENTRY entry_computation {
  x = s32[2,3] parameter(0)
  y = s32[3,2] parameter(1)
  transpose = s32[2,3] transpose(y), dimensions={1,0}
  ROOT dot = s32[2] dot(x, transpose), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_contracting_dims={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TransposeFoldingTest, FoldTransposeOfBatchWhenPermitted) {
  constexpr std::string_view kHloString = R"(
HloModule FoldDotTranspose

ENTRY entry_computation {
  x = s32[5,2,3] parameter(0)
  y = s32[3,5,4] parameter(1)
  transpose = s32[5,3,4] transpose(y), dimensions={1,0,2}
  ROOT dot = s32[5,2,4] dot(x, transpose), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  TransposeFolding transpose_folding(
      /*dot_can_fold_transpose_operand=*/[](const HloInstruction&, int64_t) {
        return true;
      });
  EXPECT_THAT(transpose_folding.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Parameter(0), op::Parameter(1),
                      /*lhs_contracting_dim=*/2, /*rhs_contracting_dim=*/0));
}

TEST_F(TransposeFoldingTest, DontFoldTransposeOfRank1Dot) {
  constexpr std::string_view kHloString = R"(
HloModule FoldDotTranspose

ENTRY entry_computation {
  x = s32[3] parameter(0)
  y = s32[3,2] parameter(1)
  transpose = s32[2,3] transpose(y), dimensions={1,0}
  ROOT dot = s32[2] dot(x, transpose), lhs_batch_dims={}, rhs_batch_dims={}, lhs_contracting_dims={0}, rhs_contracting_dims={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TransposeFoldingTest, DontFoldTransposeOfDotWithoutContractingDims) {
  constexpr std::string_view kHloString = R"(
HloModule FoldDotTranspose

ENTRY entry_computation {
  x = s32[3,4] parameter(0)
  y = s32[3,4,6,7] parameter(1)
  transpose = s32[3,4,7,6] transpose(y), dimensions={0,1,3,2}
  ROOT dot = s32[3,4,7,6] dot(x, transpose), lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={}, rhs_contracting_dims={}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TransposeFoldingTest, FoldDotTransposeConstant) {
  constexpr std::string_view kHloString = R"(
HloModule FoldDotTransposeConstant

ENTRY entry_computation {
  constant = s32[2,1]{1,0} constant({ { 1 }, { 2 } })
  transpose = s32[1,2]{1,0} transpose(constant), dimensions={1,0}
  constant.1 = s32[3,2]{1,0} constant({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  transpose.1 = s32[2,3]{1,0} transpose(constant.1), dimensions={1,0}
  ROOT dot = s32[1,3]{1,0} dot(transpose, transpose.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Constant(), op::Constant(),
                      /*lhs_contracting_dim=*/0, /*rhs_contracting_dim=*/1));
}

TEST_F(TransposeFoldingTest, FoldDotTransposeInCall) {
  constexpr std::string_view kHloString = R"(
HloModule FoldDotTransposeInCall

callee {
  name.0 = s32[2,3]{1,0} parameter(0)
  name.1 = s32[2,3]{1,0} parameter(1)
  transpose.clone = s32[3,2]{1,0} transpose(name.0), dimensions={1,0}
  ROOT dot.clone = s32[2,2]{1,0} dot(name.1, transpose.clone), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry_computation {
  y = s32[2,3]{1,0} parameter(1)
  x = s32[2,3]{1,0} parameter(0)
  ROOT call = s32[2,2]{1,0} call(y, x), to_apply=callee
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));

  const HloComputation* callee = module->GetComputationWithName("callee");
  ASSERT_NE(callee, nullptr);
  EXPECT_THAT(callee->root_instruction(),
              op::Dot(op::Parameter(1), op::Parameter(0),
                      /*lhs_contracting_dim=*/1, /*rhs_contracting_dim=*/1));
}

TEST_F(TransposeFoldingTest, FoldBatchDotTranspose) {
  constexpr std::string_view kHloString = R"(
HloModule FoldBatchDotTranspose

ENTRY entry_computation {
  x = s32[7,7,2,3]{3,2,1,0} parameter(0)
  y = s32[7,7,2,3]{3,2,1,0} parameter(1)
  transpose = s32[7,7,3,2]{3,2,1,0} transpose(y), dimensions={0,1,3,2}
  ROOT dot = s32[7,7,2,2]{3,2,1,0} dot(x, transpose), lhs_contracting_dims={3},
            rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Parameter(0), op::Parameter(1),
                      /*lhs_contracting_dim=*/3, /*rhs_contracting_dim=*/3));
}

TEST_F(TransposeFoldingTest, NoFoldBatchDotTransposeBatch) {
  constexpr std::string_view kHloString = R"(
HloModule NoFoldBatchDotTransposeBatch

ENTRY entry_computation {
  x = s32[7,7,2,3]{3,2,1,0} parameter(0)
  y = s32[7,7,2,3]{3,2,1,0} parameter(1)
  transpose = s32[7,7,3,2]{3,2,1,0} transpose(y), dimensions={1,0,3,2}
  ROOT dot = s32[7,7,2,2]{3,2,1,0} dot(x, transpose), lhs_contracting_dims={3},
            rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TransposeFoldingTest, FoldBatchDotTransposeNonContiguousBatch) {
  constexpr std::string_view kHloString = R"(
HloModule FoldBatchDotTransposeNonContiguousBatch

ENTRY entry_computation {
  x = s32[7,2,7,3]{3,2,1,0} parameter(0)
  y = s32[7,2,7,3]{3,2,1,0} parameter(1)
  transpose = s32[7,3,7,2]{3,2,1,0} transpose(y), dimensions={0,3,2,1}
  ROOT dot = s32[7,7,2,2]{3,2,1,0} dot(x, transpose), lhs_contracting_dims={3},
            rhs_contracting_dims={1}, lhs_batch_dims={0,2}, rhs_batch_dims={0,2}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::Parameter(0), op::Parameter(1),
                      /*lhs_contracting_dim=*/3, /*rhs_contracting_dim=*/3));
}

TEST_F(TransposeFoldingTest, NoFoldBatchDotTransposeIdentity) {
  constexpr std::string_view kHloString = R"(
HloModule NoFoldBatchDotTransposeIdentity

ENTRY entry_computation {
  x = s32[7,7,2,3]{3,2,1,0} parameter(0)
  y = s32[7,7,3,2]{3,2,1,0} parameter(1)
  transpose = s32[7,7,3,2]{3,2,1,0} transpose(y), dimensions={0,1,2,3}
  ROOT dot = s32[7,7,2,2]{3,2,1,0} dot(x, transpose), lhs_contracting_dims={3},
            rhs_contracting_dims={2}, lhs_batch_dims={0,1}, rhs_batch_dims={0,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(false));
}

TEST_F(TransposeFoldingTest, FoldTransposeWithBackendConfig) {
  constexpr std::string_view kHloString = R"(
HloModule FoldTransposeWithBackendConfig

ENTRY entry_computation {
  x = s32[7,2,7,3]{3,2,1,0} parameter(0)
  y = s32[7,2,7,3]{3,2,1,0} parameter(1)
  transpose = s32[7,3,7,2]{3,2,1,0} transpose(y), dimensions={0,3,2,1}
  ROOT dot = s32[7,7,2,2]{3,2,1,0} dot(x, transpose), lhs_contracting_dims={3},
            rhs_contracting_dims={1}, lhs_batch_dims={0,2}, rhs_batch_dims={0,2}, backend_config={"force_earliest_schedule":false}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_THAT(TransposeFolding().Run(module.get()), IsOkAndHolds(true));
  EXPECT_TRUE(
      module->entry_computation()->root_instruction()->has_backend_config());
}

}  // namespace
}  // namespace zkx
