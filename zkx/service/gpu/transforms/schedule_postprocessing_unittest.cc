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

#include "zkx/service/gpu/transforms/schedule_postprocessing.h"

#include <memory>

#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace zkx::gpu {
namespace {

using SchedulePostprocessingTest = HloHardwareIndependentTestBase;

TEST_F(SchedulePostprocessingTest, SynchronousOpsNotChanged) {
  constexpr std::string_view kHloString = R"(
  HloModule module, is_scheduled=true

  ENTRY entry {
    pu32 = u32[1] parameter(0)

    all-gather-start = (u32[1], u32[2]) all-gather-start(pu32), dimensions={0}, backend_config={"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false}}
    ROOT all-gather-done = u32[2] all-gather-done(all-gather-start)
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kHloString)));
  SchedulePostprocessing pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SchedulePostprocessingTest, P2POpsNotChanged) {
  constexpr std::string_view kHloString = R"(
  HloModule module, is_scheduled=true

  ENTRY main {
    f0 = u32[] constant(0)
    init = u32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all = token[] after-all()
    recv = (u32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=2,
      frontend_attributes={
      _zkx_send_recv_source_target_pairs="{{0,1}, {1,2}}"
    }
    recv-done = (u32[1, 1024, 1024], token[]) recv-done(recv), channel_id=2
    ROOT recv-data = u32[1, 1024, 1024] get-tuple-element(recv-done), index=0
  }
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule((kHloString)));
  SchedulePostprocessing pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: HloInstruction::CreateCustomCall
// TEST_F(SchedulePostprocessingTest, AsynchronousOpsChanged) {
//   constexpr std::string_view kHloString = R"(
//   HloModule module, is_scheduled=true

//   ENTRY entry {
//     pu32 = u32[1] parameter(0)
//     pu32.2 = u32[1] custom-call(pu32), custom_call_target="my_custom_call"
//     all-gather-start = (u32[1], u32[2]) all-gather-start(pu32.2), dimensions={0}, backend_config={"collective_backend_config":{"is_sync":false}}
//     ROOT all-gather-done = u32[2] all-gather-done(all-gather-start)
//   }
// )";
//   TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
//                           ParseAndReturnUnverifiedModule((kHloString)));
//   SchedulePostprocessing pass;
//   TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
//   EXPECT_TRUE(changed);

//   HloInstruction* start = FindInstruction(module.get(), "all-gather-start");
//   TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
//                           start->backend_config<GpuBackendConfig>());
//   const CollectiveBackendConfig& collective_backend_config =
//       gpu_config.collective_backend_config();
//   EXPECT_TRUE(collective_backend_config.no_parallel_custom_call());
// }

// TEST_F(SchedulePostprocessingTest, AsynchronousOpsWithParallelCustomcall) {
//   constexpr std::string_view kHloString = R"(
//   HloModule module, is_scheduled=true

//   ENTRY entry {
//     pu32 = u32[1] parameter(0)
//     all-gather-start = (u32[1], u32[2]) all-gather-start(pu32), dimensions={0}, backend_config={"collective_backend_config":{"is_sync":false}}
//     pu32.2 = u32[1] custom-call(pu32), custom_call_target="my_custom_call"
//     all-gather-done = u32[2] all-gather-done(all-gather-start)
//     ROOT out = (u32[1], u32[2]) tuple(u32[1] pu32.2, u32[2] all-gather-done)
//   }
// )";
//   TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
//                           ParseAndReturnUnverifiedModule((kHloString)));
//   SchedulePostprocessing pass;
//   TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
//   EXPECT_FALSE(changed);

//   HloInstruction* start = FindInstruction(module.get(), "all-gather-start");
//   TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
//                           start->backend_config<GpuBackendConfig>());
//   const CollectiveBackendConfig& collective_backend_config =
//       gpu_config.collective_backend_config();
//   EXPECT_FALSE(collective_backend_config.no_parallel_custom_call());
// }
// clang-format on

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: HloInstruction::CreateCustomCall
// TEST_F(SchedulePostprocessingTest,
//        AsynchronousOpsWithParallelNestedCustomcall) {
//   constexpr std::string_view kHloString = R"(
//   HloModule module, is_scheduled=true
//   foo {
//     v = u32[1] parameter(0)
//     ROOT ret = u32[1] custom-call(v), custom_call_target="my_custom_call"
//   }

//   ENTRY entry {
//     pu32 = u32[1] parameter(0)
//     all-gather-start = (u32[1], u32[2]) all-gather-start(pu32), dimensions={0}, backend_config={"collective_backend_config":{"is_sync":false}}
//     pu32.2 = u32[1] call(u32[1] pu32), to_apply=foo
//     all-gather-done = u32[2] all-gather-done(all-gather-start)
//     ROOT out = (u32[1], u32[2]) tuple(u32[1] pu32.2, u32[2] all-gather-done)
//   }
// )";
//   TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
//                           ParseAndReturnUnverifiedModule((kHloString)));
//   SchedulePostprocessing pass;
//   TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
//   EXPECT_FALSE(changed);

//   HloInstruction* start = FindInstruction(module.get(), "all-gather-start");
//   TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
//                           start->backend_config<GpuBackendConfig>());
//   const CollectiveBackendConfig& collective_backend_config =
//       gpu_config.collective_backend_config();
//   EXPECT_FALSE(collective_backend_config.no_parallel_custom_call());
// }
// clang-format on

}  // namespace
}  // namespace zkx::gpu
