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

#ifndef ZKX_SERVICE_GPU_GPU_DEVICE_INFO_FOR_TESTS_H_
#define ZKX_SERVICE_GPU_GPU_DEVICE_INFO_FOR_TESTS_H_

#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {

class TestGpuDeviceInfo {
 public:
  static se::DeviceDescription RTXA6000DeviceInfo(
      se::GpuComputeCapability cc = se::CudaComputeCapability(8, 9));
  static se::DeviceDescription AMDMI210DeviceInfo();
  // Returns default RTXA6000 or AMDMI210 device info
  static se::DeviceDescription CudaOrRocmDeviceInfo();
};

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_GPU_DEVICE_INFO_FOR_TESTS_H_
