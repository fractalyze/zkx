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

#ifndef ZKX_PYTHON_PJRT_IFRT_PJRT_EXECUTABLE_H_
#define ZKX_PYTHON_PJRT_IFRT_PJRT_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"

#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_sharding.h"
#include "zkx/pjrt/pjrt_client.h"
#include "zkx/pjrt/pjrt_executable.h"
#include "zkx/pjrt/pjrt_layout.h"
#include "zkx/python/ifrt/array.h"
#include "zkx/python/ifrt/attribute_map.h"
#include "zkx/python/ifrt/device.h"
#include "zkx/python/ifrt/device_list.h"
#include "zkx/python/ifrt/dtype.h"
#include "zkx/python/ifrt/executable.h"
#include "zkx/python/ifrt/future.h"
#include "zkx/python/ifrt/host_callback.h"
#include "zkx/python/ifrt/shape.h"
#include "zkx/python/ifrt/sharding.h"
#include "zkx/python/ifrt/user_context.h"
#include "zkx/python/pjrt_ifrt/pjrt_attribute_map_util.h"
#include "zkx/python/pjrt_ifrt/pjrt_client.h"
#include "zkx/python/pjrt_ifrt/pjrt_host_callback.h"
#include "zkx/util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx::ifrt {

// PjRt-compatible `Executable` interface.
class PjRtCompatibleExecutable
    : public llvm::RTTIExtends<PjRtCompatibleExecutable, Executable> {
 public:
  // APIs that allow direct access to `zkx::PjRtExecutable` for PjRt-only
  // operations.
  virtual zkx::PjRtExecutable* pjrt_executable() = 0;

  static char ID;
};

// PjRt-compatible `LoadedExecutable` interface.
class PjRtCompatibleLoadedExecutable
    : public llvm::RTTIExtends<PjRtCompatibleLoadedExecutable,
                               LoadedExecutable> {
 public:
  // Key for the call location attribute in the custom_options attribute map.
  static constexpr std::string_view kCallLocation = "call_location";

  // APIs that allow direct access to `zkx::PjRtLoadedExecutable` for PjRt-only
  // operations.
  virtual zkx::PjRtLoadedExecutable* pjrt_loaded_executable() = 0;
  virtual std::shared_ptr<zkx::PjRtLoadedExecutable>
  shared_ptr_pjrt_loaded_executable() = 0;

  static char ID;
};

// `Executable` implementation that wraps a `zkx::PjRtExecutable`.
class PjRtExecutable final
    : public llvm::RTTIExtends<PjRtExecutable, PjRtCompatibleExecutable> {
 public:
  // Creates PjRtExecutable from zkx::PjRtExecutable.
  static absl::StatusOr<ExecutableRef> Create(
      std::shared_ptr<zkx::PjRtExecutable> pjrt_executable);

  // PjRtCompatibleExecutable implementation.

  zkx::PjRtExecutable* pjrt_executable() override {
    DCHECK(this);
    return pjrt_executable_.get();
  }

  // Executable implementation.

  ~PjRtExecutable() override = default;

  std::string_view name() const override {
    DCHECK(this);
    return pjrt_executable_->name();
  }

  std::optional<std::vector<OpSharding>> GetParameterShardings()
      const override {
    DCHECK(this);
    return pjrt_executable_->GetParameterShardings();
  }

  std::optional<std::vector<OpSharding>> GetOutputShardings() const override {
    DCHECK(this);
    return pjrt_executable_->GetOutputShardings();
  }

  absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
  GetParameterLayouts() const override {
    DCHECK(this);
    return pjrt_executable_->GetParameterLayouts();
  }

  absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
  GetOutputLayouts() const override {
    // TODO(hyeontaek): Return `output_layouts_` instead, which can distinguish
    // between default and custom layouts, once the users of
    // `GetOutputLayouts()` understand `nullptr` elements.
    DCHECK(this);
    return pjrt_executable_->GetOutputLayouts();
  }

  absl::StatusOr<std::optional<std::string>> Fingerprint() const override;

  absl::StatusOr<std::string> Serialize() const override;

  int num_devices() const override {
    DCHECK(this);
    return pjrt_executable_->num_replicas() *
           pjrt_executable_->num_partitions();
  }
  int64_t SizeOfGeneratedCodeInBytes() const override {
    DCHECK(this);
    return pjrt_executable_->SizeOfGeneratedCodeInBytes();
  }
  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    DCHECK(this);
    return pjrt_executable_->GetCompiledMemoryStats();
  }

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    DCHECK(this);
    return pjrt_executable_->GetHloModules();
  }

  absl::StatusOr<zkx::ifrt::AttributeMap> GetCostAnalysis() const override {
    TF_ASSIGN_OR_RETURN(auto result, pjrt_executable_->GetCostAnalysis());
    return zkx::ifrt::FromPjRtAttributeMap(std::move(result));
  }

  absl::StatusOr<std::vector<std::vector<std::string_view>>>
  GetOutputMemoryKinds() const override {
    return pjrt_executable_->GetOutputMemoryKinds();
  }

  static char ID;

 protected:
  explicit PjRtExecutable(std::shared_ptr<zkx::PjRtExecutable> pjrt_executable)
      : pjrt_executable_(std::move(pjrt_executable)) {}

  std::shared_ptr<zkx::PjRtExecutable> pjrt_executable_;
};

// `LoadedExecutable` implementation that wraps a `zkx::PjRtLoadedExecutable`.
class PjRtLoadedExecutable final
    : public llvm::RTTIExtends<PjRtLoadedExecutable,
                               PjRtCompatibleLoadedExecutable> {
 public:
  using LoadedExecutable::ExecuteOptions;
  using LoadedExecutable::ExecuteResult;

  // Creates PjRtExecutable from zkx::PjRtLoadedExecutable. We expect that
  // zkx::PjRtLoadedExecutable has fixed output dtypes/shapes/shardings.
  // PjRtLoadedExecutable::GetHloModules() must be implemented.
  static absl::StatusOr<LoadedExecutableRef> Create(
      PjRtClient* client,
      std::shared_ptr<zkx::PjRtLoadedExecutable> pjrt_loaded_executable,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks,
      DeviceListRef executable_devices);

  // Creates PjRtExecutable from an MHLO or StableHLO MLIR module. We expect
  // that zkx::PjRtLoadedExecutable has fixed output dtypes/shapes/shardings. If
  // options.executable_build_options has use_auto_spmd_partitioning or
  // allow_spmd_sharding_propagation_to_output enabled,
  // PjRtLoadedExecutable::GetHloModules() must be implemented.
  static absl::StatusOr<LoadedExecutableRef> Create(
      PjRtClient* client, mlir::ModuleOp module,
      zkx::CompileOptions compile_options,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks,
      DeviceListRef executable_devices);

  // PjRtCompatibleLoadedExecutable implementation.

  zkx::PjRtLoadedExecutable* pjrt_loaded_executable() override {
    DCHECK(this);
    return pjrt_loaded_executable_.get();
  }
  std::shared_ptr<zkx::PjRtLoadedExecutable> shared_ptr_pjrt_loaded_executable()
      override {
    DCHECK(this);
    return pjrt_loaded_executable_;
  }

  // LoadedExecutable implementation.

  ~PjRtLoadedExecutable() override;

  std::string_view name() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->name();
  }

  absl::StatusOr<absl::Span<const int>> GetDonatableInputIndices()
      const override {
    return absl::UnimplementedError(
        "PjRtLoadedExecutable::GetDonatableInputIndices is not implemented.");
  }

  UserContextRef user_context() const override { return user_context_; }

  Future<> GetReadyFuture() const override {
    // PjRtCompiler blocks until compilation finishes and returns only the
    // executables that are ready.
    return Future<>(absl::OkStatus());
  }

  std::optional<std::vector<OpSharding>> GetParameterShardings()
      const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetParameterShardings();
  }

  std::optional<std::vector<OpSharding>> GetOutputShardings() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetOutputShardings();
  }

  absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
  GetParameterLayouts() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetParameterLayouts();
  }

  absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
  GetOutputLayouts() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetOutputLayouts();
  }

  absl::StatusOr<std::optional<std::string>> Fingerprint() const override;

  absl::StatusOr<std::unique_ptr<ExecutableVersion>> executable_version()
      const override {
    return absl::UnimplementedError("Not implemented");
  }

  absl::StatusOr<std::string> Serialize() const override;

  absl::StatusOr<std::string> GetHumanReadableProgramText() const override {
    TF_ASSIGN_OR_RETURN(auto hlo_modules,
                        pjrt_loaded_executable_->GetHloModules());
    return absl::StrJoin(hlo_modules, "\n\n",
                         [](std::string* out, const auto& hlo_module) {
                           absl::StrAppend(out, hlo_module->ToString());
                         });
  }

  int num_devices() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->num_replicas() *
           pjrt_loaded_executable_->num_partitions();
  }
  int64_t SizeOfGeneratedCodeInBytes() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->SizeOfGeneratedCodeInBytes();
  }
  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetCompiledMemoryStats();
  }

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetHloModules();
  }

  absl::StatusOr<std::vector<std::vector<std::string_view>>>
  GetOutputMemoryKinds() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetOutputMemoryKinds();
  }

  PjRtClient* client() const override {
    DCHECK(this);
    return client_;
  }
  absl::StatusOr<ExecuteResult> Execute(
      absl::Span<ArrayRef> args, const ExecuteOptions& options,
      std::optional<DeviceListRef> devices) override;

  const DeviceListRef& devices() const override { return devices_; }

  absl::Span<Device* const> addressable_devices() const override {
    DCHECK(this);
    return addressable_devices_;
  }

  absl::StatusOr<zkx::ifrt::AttributeMap> GetCostAnalysis() const override {
    TF_ASSIGN_OR_RETURN(auto result,
                        pjrt_loaded_executable_->GetCostAnalysis());
    return zkx::ifrt::FromPjRtAttributeMap(std::move(result));
  }

  static char ID;

 private:
  static absl::StatusOr<LoadedExecutableRef> CreateInternal(
      PjRtClient* client,
      std::shared_ptr<zkx::PjRtLoadedExecutable> pjrt_loaded_executable,
      absl::Span<const zkx::PrimitiveType> result_element_types,
      absl::Span<const zkx::DimensionVector> result_dimensions,
      const std::optional<zkx::HloSharding>& result_hlo_sharding,
      const std::optional<std::vector<std::string_view>>& result_memory_kinds,
      const std::optional<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>&
          output_layouts,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks,
      DeviceListRef executable_devices);

  PjRtLoadedExecutable(
      PjRtClient* client,
      std::shared_ptr<zkx::PjRtLoadedExecutable> pjrt_loaded_executable,
      DeviceListRef devices, std::vector<Device*> addressable_devices,
      std::vector<tsl::RCReference<LoadedHostCallback>>
          all_loaded_host_callbacks,
      std::vector<PjRtHostSendAndRecvLoadedHostCallback*>
          host_send_recv_callbacks,
      std::vector<DType> output_dtypes, std::vector<Shape> output_shapes,
      std::vector<ShardingRef> output_shardings,
      std::optional<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
          output_layouts);

  PjRtClient* client_;  // not owned
  std::shared_ptr<zkx::PjRtLoadedExecutable> pjrt_loaded_executable_;
  // Devices that `pjrt_loaded_executable_` runs on. Empty if the executable is
  // portable.
  DeviceListRef devices_;
  std::vector<Device*> addressable_devices_;
  std::shared_ptr<std::vector<tsl::RCReference<LoadedHostCallback>>>
      all_loaded_host_callbacks_;
  std::vector<PjRtHostSendAndRecvLoadedHostCallback*> host_send_recv_callbacks_;

  // Output array specs. If the executable is portable, shardings in
  // `output_shardings_` will use an arbitrary addressable device, and will be
  // overridden by a `SingleDeviceSharding` generated on the fly at execution
  // time.
  std::vector<DType> output_dtypes_;
  std::vector<Shape> output_shapes_;
  std::vector<ShardingRef> output_shardings_;
  std::optional<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>
      output_layouts_;
  const UserContextRef user_context_;
};

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_PJRT_IFRT_PJRT_EXECUTABLE_H_
