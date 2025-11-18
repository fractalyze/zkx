/* Copyright 2024 The OpenXLA Authors.
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

#include "zkx/service/gpu/flag_utils.h"

#include "gtest/gtest.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/service/latency_hiding_scheduler.h"

namespace zkx::gpu {
namespace {

TEST(FlagUtilsTest, IsPassEnabledAtOptimizationEffort) {
  HloModuleConfig config;
  config.set_exec_time_optimization_effort(kExtraCollectiveOptimizations + 1);
  HloModule module("test_module", config);

  // Collective optimization passes.
  // TODO(chokobole): Uncomment this. Dependency: CollectivePipeliner
  //   EXPECT_TRUE(IsPassEnabledAtOptimizationEffort<CollectivePipeliner>(module));
  // TODO(chokobole): Uncomment this. Dependency: DoubleBufferLoopUnrolling
  //   EXPECT_TRUE(
  //       IsPassEnabledAtOptimizationEffort<DoubleBufferLoopUnrolling>(module));
  EXPECT_TRUE(
      IsPassEnabledAtOptimizationEffort<LatencyHidingScheduler>(module));

  // Other passes.
  // TODO(chokobole): Uncomment this. Dependency: HloDCE
  //   EXPECT_TRUE(IsPassEnabledAtOptimizationEffort<HloDCE>(module));

  config.set_exec_time_optimization_effort(kExtraCollectiveOptimizations - 1);
  module.set_config(config);

  // Collective optimization passes.
  // TODO(chokobole): Uncomment this. Dependency: CollectivePipeliner
  //   EXPECT_FALSE(IsPassEnabledAtOptimizationEffort<CollectivePipeliner>(module));
  // TODO(chokobole): Uncomment this. Dependency: DoubleBufferLoopUnrolling
  //   EXPECT_FALSE(
  //       IsPassEnabledAtOptimizationEffort<DoubleBufferLoopUnrolling>(module));
  EXPECT_FALSE(
      IsPassEnabledAtOptimizationEffort<LatencyHidingScheduler>(module));
}

}  // namespace
}  // namespace zkx::gpu
