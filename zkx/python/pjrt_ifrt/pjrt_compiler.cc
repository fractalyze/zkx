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

#include "zkx/python/pjrt_ifrt/pjrt_compiler.h"

#include <cstdint>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "llvm/Support/Casting.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/pjrt/pjrt_client.h"
#include "zkx/pjrt/pjrt_compiler.h"
#include "zkx/pjrt/pjrt_executable.h"
#include "zkx/python/ifrt/device.h"
#include "zkx/python/ifrt/hlo/hlo_program.h"
#include "zkx/python/pjrt_ifrt/pjrt_client.h"
#include "zkx/python/pjrt_ifrt/pjrt_executable.h"
#include "zkx/python/pjrt_ifrt/pjrt_topology.h"
#include "zkx/python/pjrt_ifrt/zkx_compiler.h"
#include "zkx/service/computation_placer.h"

namespace zkx::ifrt {

namespace {

// Translates IFRT device IDs to PjRt global device IDs in place. When an error
// occurs, `options` may have invalid device IDs.
absl::Status TranslateDeviceIds(PjRtClient* client,
                                zkx::CompileOptions& options) {
  if (options.executable_build_options.device_ordinal() != -1) {
    TF_ASSIGN_OR_RETURN(
        auto pjrt_global_device_id,
        client->GetPjRtGlobalDeviceId(
            DeviceId(options.executable_build_options.device_ordinal())));
    options.executable_build_options.set_device_ordinal(
        pjrt_global_device_id.value());
  }
  if (options.executable_build_options.has_device_assignment()) {
    absl::Status result;
    zkx::DeviceAssignment device_assignment =
        options.executable_build_options.device_assignment();
    device_assignment.Each(
        [&](int64_t replica, int64_t computation, int64_t* device_id) {
          if (!result.ok()) {
            return;
          }
          auto pjrt_global_device_id =
              client->GetPjRtGlobalDeviceId(DeviceId(*device_id));
          if (pjrt_global_device_id.ok()) {
            *device_id = pjrt_global_device_id->value();
          } else {
            result.Update(pjrt_global_device_id.status());
          }
        });
    TF_RETURN_IF_ERROR(result);
    options.executable_build_options.set_device_assignment(
        std::move(device_assignment));
  }
  return absl::OkStatus();
}

}  // namespace

char PjRtCompiler::ID = 0;

absl::StatusOr<LoadedExecutableRef> PjRtCompiler::CompileAndLoad(
    std::unique_ptr<Program> program, std::unique_ptr<CompileOptions> options) {
  DCHECK(this);
  const auto* zkx_program = llvm::dyn_cast<HloProgram>(program.get());
  if (zkx_program == nullptr) {
    return absl::InvalidArgumentError("PjRtCompiler requires an HloProgram");
  }
  TF_ASSIGN_OR_RETURN(auto zkx_compile_options,
                      GetZkxCompileOptions(std::move(options)));
  TF_RETURN_IF_ERROR(
      TranslateDeviceIds(client_, zkx_compile_options->compile_options));
  return PjRtLoadedExecutable::Create(
      client_, zkx_program->mlir_module(),
      std::move(zkx_compile_options->compile_options),
      std::move(zkx_compile_options->loaded_host_callbacks),
      std::move(zkx_compile_options->devices));
}

absl::StatusOr<ExecutableRef> PjRtCompiler::Compile(
    std::unique_ptr<Program> program, const Topology& topology,
    std::unique_ptr<CompileOptions> options) {
  DCHECK(this);
  const auto* zkx_program = llvm::dyn_cast<HloProgram>(program.get());
  if (zkx_program == nullptr) {
    return absl::InvalidArgumentError("PjRtCompiler requires an HloProgram");
  }
  TF_ASSIGN_OR_RETURN(auto zkx_compile_options,
                      GetZkxCompileOptions(std::move(options)));
  TF_RETURN_IF_ERROR(
      TranslateDeviceIds(client_, zkx_compile_options->compile_options));
  const auto* pjrt_topology = llvm::dyn_cast<PjRtTopology>(&topology);
  if (pjrt_topology == nullptr) {
    return absl::InvalidArgumentError("PjRtCompiler requires a PjRtTopology");
  }
  TF_ASSIGN_OR_RETURN(
      auto executable,
      PjRtCompile(zkx_compile_options->compile_options,
                  zkx_program->mlir_module(), *pjrt_topology->description()));
  return PjRtExecutable::Create(std::move(executable));
}

absl::StatusOr<LoadedExecutableRef> PjRtCompiler::DeserializeLoadedExecutable(
    std::string_view serialized,
    std::unique_ptr<DeserializeExecutableOptions> options) {
  DCHECK(this);
  TF_ASSIGN_OR_RETURN(auto zkx_deserialize_options,
                      GetZkxDeserializeExecutableOptions(std::move(options)));
  if (zkx_deserialize_options->compile_options.has_value()) {
    TF_RETURN_IF_ERROR(
        TranslateDeviceIds(client_, *zkx_deserialize_options->compile_options));
  }
  TF_ASSIGN_OR_RETURN(
      auto pjrt_loaded_executable,
      client_->pjrt_client()->LoadSerializedExecutable(
          serialized, std::move(zkx_deserialize_options->compile_options),
          zkx::LoadOptions()));
  // TODO(emilyaf): Remove the else branch once devices are plumbed through from
  // Australis and are always present in the DeserializeExecutableOptions.
  DeviceListRef device_list;
  if (zkx_deserialize_options->devices.has_value()) {
    device_list = std::move(zkx_deserialize_options->devices.value());
  } else {
    TF_ASSIGN_OR_RETURN(
        device_list, GetDeviceListFromDeviceAssignment(
                         client_, pjrt_loaded_executable->device_assignment()));
  }
  return PjRtLoadedExecutable::Create(
      client_,
      std::shared_ptr<zkx::PjRtLoadedExecutable>(
          std::move(pjrt_loaded_executable)),
      std::move(zkx_deserialize_options->loaded_host_callbacks),
      std::move(device_list));
}

}  // namespace zkx::ifrt
