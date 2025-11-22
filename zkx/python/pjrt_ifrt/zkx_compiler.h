/* Copyright 2023 The OpenXLA Authors.
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

#ifndef ZKX_PYTHON_PJRT_IFRT_ZKX_COMPILER_H_
#define ZKX_PYTHON_PJRT_IFRT_ZKX_COMPILER_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include "zkx/pjrt/pjrt_executable.h"
#include "zkx/python/ifrt/compiler.h"
#include "zkx/python/ifrt/device_list.h"
#include "zkx/python/ifrt/executable_serdes.h"
#include "zkx/python/ifrt/host_callback.h"

namespace zkx::ifrt {

// Wraps compilation options for a ZKX computation.
//
// TODO(hyeontaek): Move this class out of pjrt_ifrt.
//
// TODO(hyeontaek): Move `loaded_host_callbacks` to a (new) `LoadOptions`
// because compilation (without loading) should not take them.
struct ZkxCompileOptions
    : llvm::RTTIExtends<ZkxCompileOptions, CompileOptions> {
  ZkxCompileOptions() = default;
  explicit ZkxCompileOptions(zkx::CompileOptions compile_options,
                             DeviceListRef devices,
                             std::vector<tsl::RCReference<LoadedHostCallback>>
                                 loaded_host_callbacks = {})
      : compile_options(std::move(compile_options)),
        devices(std::move(devices)),
        loaded_host_callbacks(std::move(loaded_host_callbacks)) {}

  zkx::CompileOptions compile_options;
  DeviceListRef devices;
  std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks;

  // CompileOptions implementation.

  ~ZkxCompileOptions() override = default;

  static char ID;
};

// Wraps deserialization options for a ZKX computation.
//
// TODO(hyeontaek): Move this class out of pjrt_ifrt.
//
// TODO(hyeontaek): Move `loaded_host_callbacks` to a (new) `LoadOptions`
// because deserialization (without loading) should not take them.
// TODO(emilyaf): Make `devices` non-optional once it is plumbed through from
// Australis.
struct ZkxDeserializeExecutableOptions
    : llvm::RTTIExtends<ZkxDeserializeExecutableOptions,
                        DeserializeExecutableOptions> {
  ZkxDeserializeExecutableOptions() = default;
  explicit ZkxDeserializeExecutableOptions(
      std::optional<zkx::CompileOptions> compile_options,
      std::optional<DeviceListRef> devices,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks =
          {})
      : llvm::RTTIExtends<ZkxDeserializeExecutableOptions,
                          DeserializeExecutableOptions>(std::move(devices)),
        compile_options(std::move(compile_options)),
        loaded_host_callbacks(std::move(loaded_host_callbacks)) {}

  // `compile_options` may be unspecified if deserialization does not override
  // it.
  std::optional<zkx::CompileOptions> compile_options;
  std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks;

  // DeserializeExecutableOptions implementation.

  ~ZkxDeserializeExecutableOptions() override = default;

  static char ID;
};

// Gets `zkx::ifrt::ZkxCompileOptions` from `zkx::ifrt::CompileOptions`.
absl::StatusOr<std::unique_ptr<ZkxCompileOptions>> GetZkxCompileOptions(
    std::unique_ptr<CompileOptions> options);

// Gets `zkx::ifrt::ZkxDeserializeExecutableOptions` from
// `zkx::ifrt::DeserializeExecutableOptions`.
absl::StatusOr<std::unique_ptr<ZkxDeserializeExecutableOptions>>
GetZkxDeserializeExecutableOptions(
    std::unique_ptr<DeserializeExecutableOptions> options);

// Gets `zkx::ifrt::DeviceListRef` from `zkx::DeviceAssignment`.
absl::StatusOr<zkx::ifrt::DeviceListRef> GetDeviceListFromDeviceAssignment(
    zkx::ifrt::Client* ifrt_client,
    const zkx::DeviceAssignment& device_assignment);

// Gets `zkx::ifrt::DeviceListRef` from `zkx::ZkxCompileOptions`.
absl::StatusOr<zkx::ifrt::DeviceListRef> GetDeviceListFromZkxCompileOptions(
    zkx::ifrt::Client* ifrt_client, const zkx::CompileOptions& compile_options);

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_PJRT_IFRT_ZKX_COMPILER_H_
