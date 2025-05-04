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

#include "zkx/service/cpu/cpu_runtime.h"

#include "gtest/gtest.h"

#include "zkx/service/cpu/runtime_custom_call_status.h"
#include "zkx/service/custom_call_status_internal.h"

namespace zkx::cpu::runtime {

TEST(CpuRuntimeTest, SuccessStatus) {
  ZkxCustomCallStatus success_status;
  // Success is the default state.
  ASSERT_TRUE(__zkx_cpu_runtime_StatusIsSuccess(&success_status));
}

TEST(CpuRuntimeTest, FailureStatus) {
  ZkxCustomCallStatus success_status;
  ZkxCustomCallStatusSetFailure(&success_status, "Failed", 6);
  ASSERT_FALSE(__zkx_cpu_runtime_StatusIsSuccess(&success_status));
}

// When run_options is null, the process should not crash and the device ordinal
// should be 0.
TEST(CpuRuntimeTest, GetDeviceOrdinalWhenRunOptionsEmpty) {
  EXPECT_EQ(GetDeviceOrdinal(/*run_options=*/nullptr), 0);
}

// When the device ordinal is set directly in run options, it should be returned
// (and NOT the value from stream).
TEST(CpuRuntimeTest, GetDeviceOrdinalWhenSetInRunOptions) {
  // GetDeviceOrdinal implementation bases on the fact that device ordinal is
  // -1 by default. So we need to assert for that here to avoid crash in case
  // the default value changes in the future.
  ExecutableRunOptions run_options;
  ASSERT_EQ(run_options.device_ordinal(), -1);

  // Actual test - set device ordinal in run options and check that it is
  // returned.
  run_options.set_device_ordinal(3);
  EXPECT_EQ(GetDeviceOrdinal(&run_options), 3);
}

// TODO(abanas): Add test case for the device ordinal with stream case. It
// requires mocking the stream and stream executor.

}  // namespace zkx::cpu::runtime
