/* Copyright 2023 The OpenXLA Authors.

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

#include "zkx/service/gpu/transforms/async_collective_annotator.h"

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/hlo/utils/hlo_query.h"
#include "zkx/service/gpu/backend_configs.pb.h"
#include "zkx/util.h"

namespace zkx::gpu {
namespace {

constexpr std::string_view kHloString = R"(
  HloModule ModuleWithAsync

  addu64 {
    p0 = u64[] parameter(0)
    p1 = u64[] parameter(1)
    ROOT add = u64[] add(p0, p1)
  }

  addu32 {
    p0 = u32[] parameter(0)
    p1 = u32[] parameter(1)
    ROOT add = u32[] add(p0, p1)
  }

  reduce_scatteru64 {
    p0 = u64[2] parameter(0)
    ROOT result = u64[1] reduce-scatter(p0), replica_groups={},
                      dimensions={0}, to_apply=addu64
  }

  ENTRY entry {
    pu64 = u64[1] parameter(0)
    pu32 = u32[1] parameter(1)

    aru64-start = u64[1] all-reduce-start(pu64), to_apply=addu64
    aru64-done = u64[1] all-reduce-done(aru64-start)

    aru32-start = u32[1] all-reduce-start(pu32), to_apply=addu32
    aru32-done = u32[1] all-reduce-done(aru32-start)

    agu64-start = (u64[1], u64[2]) all-gather-start(pu64), dimensions={0}
    agu64-done = u64[2] all-gather-done(agu64-start)

    agu32-start = (u32[1], u32[2]) all-gather-start(pu32), dimensions={0}
    agu32-done = u32[2] all-gather-done(agu32-start)

    cpu64-start = (u64[1], u64[1], u32[], u32[]) collective-permute-start(pu64),
                    source_target_pairs={{0,1}, {1,0}}
    cpu64-done = u64[1] collective-permute-done(cpu64-start)

    cpu32-start = (u32[1], u32[1], u32[], u32[]) collective-permute-start(pu32),
                    source_target_pairs={{0,1}, {1,0}}
    cpu32-done = u32[1] collective-permute-done(cpu32-start)

    rsu64-start = ((u64[2]), u64[1]) async-start(agu64-done), calls=reduce_scatteru64
    rsu64-done = u64[1] async-done(rsu64-start), calls=reduce_scatteru64

    ROOT tuple = (u64[1], u32[1], u64[2], u32[2], u64[1], u32[1], u64[1])
                tuple(aru64-done, aru32-done, agu64-done, agu32-done, cpu64-done,
                      cpu32-done, rsu64-done)
  }
)";

struct TestCase {
  std::string test_name;
  HloPredicate is_async_predicate;
  absl::flat_hash_set<std::string_view> expected_async;
  absl::flat_hash_set<std::string_view> expected_sync;
};

class AsyncCollectiveAnnotatorTest
    : public HloHardwareIndependentTestBase,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(AsyncCollectiveAnnotatorTest, Test) {
  const TestCase& test_case = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(kHloString, /*replica_count=*/2));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      AsyncCollectiveAnnotator(test_case.is_async_predicate).Run(module.get()));
  EXPECT_TRUE(changed);

  // Assert that all async collectives are annotated with the backend config.
  for (const HloInstruction* hlo :
       module->entry_computation()->instructions()) {
    if (!hlo_query::IsAsyncCollectiveStartOp(hlo)) {
      continue;
    }
    auto gpu_config = hlo->backend_config<GpuBackendConfig>();
    ASSERT_TRUE(gpu_config.ok());

    const CollectiveBackendConfig& backend_config =
        gpu_config.value().collective_backend_config();
    if (test_case.expected_async.contains(hlo->name())) {
      EXPECT_FALSE(backend_config.is_sync());
    }

    if (test_case.expected_sync.contains(hlo->name())) {
      EXPECT_TRUE(backend_config.is_sync());
    }
  }
}

std::vector<TestCase> TestCases() {
  HloPredicate is_u32 = [](const HloInstruction* hlo) {
    return hlo->operand(0)->shape().element_type() == PrimitiveType::U32;
  };

  return {
      {"all_async",
       HloPredicateTrue, /*expected_async=*/
       {"aru64-start", "aru32-start", "agu64-start", "agu32-start",
        "cpu64-start", "cpu32-start", "rsu64-start"},
       /*expected_sync=*/{}},
      {"all_sync",
       HloPredicateFalse,
       /*expected_async=*/{},
       /*expected_sync=*/
       {"aru64-start", "aru32-start", "agu64-start", "agu32-start",
        "cpu64-start", "cpu32-start", "rsu64-start"}},
      {"ar_async",
       HloPredicateIsOp<HloOpcode::kAllReduceStart>,
       /*expected_async=*/
       {"aru64-start", "aru32-start"},
       /*expected_sync=*/
       {"agu64-start", "agu32-start", "cpu64-start", "cpu32-start",
        "rsu64-start"}},
      {"cp_async",
       HloPredicateIsOp<HloOpcode::kCollectivePermuteStart>,
       /*expected_async=*/
       {"cpu64-start", "cpu32-start"},
       /*expected_sync=*/
       {"aru64-start", "aru32-start", "agu64-start", "agu32-start",
        "rsu64-start"}},
      {"u32_async",
       is_u32,
       /*expected_async=*/{"aru32-start", "agu32-start", "cpu32-start"},
       /*expected_sync=*/
       {"aru64-start", "agu64-start", "cpu64-start", "rsu64-start"}},
  };
}

std::string TestCaseName(const ::testing::TestParamInfo<TestCase>& test_case) {
  return test_case.param.test_name;
}

INSTANTIATE_TEST_SUITE_P(AsyncCollectiveAnnotatorTest,
                         AsyncCollectiveAnnotatorTest,
                         ::testing::ValuesIn(TestCases()), TestCaseName);
}  // namespace
}  // namespace zkx::gpu
