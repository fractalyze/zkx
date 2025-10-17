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

#ifndef ZKX_PYTHON_IFRT_MOCK_H_
#define ZKX_PYTHON_IFRT_MOCK_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/pjrt/pjrt_executable.h"
#include "zkx/pjrt/pjrt_layout.h"
#include "zkx/python/ifrt/array.h"
#include "zkx/python/ifrt/array_spec.h"
#include "zkx/python/ifrt/attribute_map.h"
#include "zkx/python/ifrt/basic_device_list.h"
#include "zkx/python/ifrt/client.h"
#include "zkx/python/ifrt/compiler.h"
#include "zkx/python/ifrt/device.h"
#include "zkx/python/ifrt/device_list.h"
#include "zkx/python/ifrt/dtype.h"
#include "zkx/python/ifrt/executable.h"
#include "zkx/python/ifrt/executable_serdes.h"
#include "zkx/python/ifrt/future.h"
#include "zkx/python/ifrt/host_callback.h"
#include "zkx/python/ifrt/index_domain.h"
#include "zkx/python/ifrt/layout.h"
#include "zkx/python/ifrt/memory.h"
#include "zkx/python/ifrt/program.h"
#include "zkx/python/ifrt/remap_plan.h"
#include "zkx/python/ifrt/shape.h"
#include "zkx/python/ifrt/sharding.h"
#include "zkx/python/ifrt/topology.h"
#include "zkx/python/ifrt/tuple.h"
#include "zkx/python/ifrt/user_context.h"
#include "zkx/python/ifrt/value.h"
#include "zkx/zkx_data.pb.h"

namespace zkx::ifrt {

// array.h

class MockArray : public llvm::RTTIExtends<MockArray, Array> {
 public:
  MockArray() = default;
  explicit MockArray(ArrayRef delegated);

  // LINT.IfChange
  MOCK_METHOD(Client*, client, (), (const, final));
  MOCK_METHOD(Future<>, GetReadyFuture, (), (const, final));
  MOCK_METHOD(Future<>, Delete, (), (final));
  MOCK_METHOD(bool, IsDeleted, (), (const, final));

  MOCK_METHOD(DType, dtype, (), (const, final));
  MOCK_METHOD(const Shape&, shape, (), (const, final));
  MOCK_METHOD(const Sharding&, sharding, (), (const, final));
  MOCK_METHOD(ShardingRef, shared_ptr_sharding, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::shared_ptr<const zkx::PjRtLayout>>,
              pjrt_layout, (), (const, final));
  MOCK_METHOD(CustomLayoutRef, layout, (), (const, final));
  MOCK_METHOD(UserContextRef, user_context, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<ArrayRef>>,
              DisassembleIntoSingleDeviceArrays,
              (ArrayCopySemantics array_copy_semantics,
               SingleDeviceShardSemantics single_device_shard_semantics),
              (final));
  MOCK_METHOD(absl::StatusOr<ArrayRef>, FullyReplicatedShard,
              (ArrayCopySemantics semantics), (final));
  MOCK_METHOD(Future<>, CopyToHostBuffer,
              (void* data,
               std::optional<absl::Span<const int64_t>> byte_strides,
               ArrayCopySemantics semantics),
              (final));
  // LINT.ThenChange(mock.cc:MockArrayDelegation)

  ArrayRef delegated() const { return delegated_; }

  std::string DebugString() const final { return "MockArray"; }

  static char ID;

 private:
  const ArrayRef delegated_;
};

// client.h

class MockClient : public llvm::RTTIExtends<MockClient, Client> {
 public:
  MockClient() = default;
  explicit MockClient(std::unique_ptr<Client> delegated);

  // LINT.IfChange
  MOCK_METHOD(absl::StatusOr<ArrayRef>, MakeArrayFromHostBuffer,
              (const void* data, DType dtype, Shape shape,
               std::optional<absl::Span<const int64_t>> byte_strides,
               ShardingRef sharding, HostBufferSemantics semantics,
               std::function<void()> on_done_with_host_buffer),
              (final));
  MOCK_METHOD(absl::StatusOr<std::vector<ArrayRef>>,
              MakeArraysFromHostBufferShards,
              (absl::Span<MakeArraysFromHostBufferShardsSpec> specs,
               HostBufferSemantics semantics),
              (final));
  MOCK_METHOD(absl::StatusOr<std::vector<ArrayRef>>, MakeErrorArrays,
              (const absl::Status& error,
               absl::Span<const ArraySpec> array_specs),
              (final));
  MOCK_METHOD(absl::StatusOr<ArrayRef>, AssembleArrayFromSingleDeviceArrays,
              (DType dtype, Shape shape, ShardingRef sharding,
               absl::Span<ArrayRef> arrays,
               ArrayCopySemantics array_copy_semantics,
               SingleDeviceShardSemantics single_device_shard_semantics),
              (final));
  MOCK_METHOD(absl::StatusOr<std::vector<ArrayRef>>, CopyArrays,
              (absl::Span<ArrayRef> arrays,
               std::optional<DeviceListRef> devices,
               std::optional<MemoryKind> memory_kind,
               ArrayCopySemantics semantics),
              (final));
  MOCK_METHOD(absl::StatusOr<std::vector<ArrayRef>>, RemapArrays,
              (const RemapPlan& plan, absl::Span<ArrayRef> arrays,
               ArrayCopySemantics semantics),
              (final));
  MOCK_METHOD(absl::StatusOr<std::vector<ArrayRef>>, ReshardArrays,
              (absl::Span<ArrayRef> arrays, absl::Span<const ArraySpec> specs,
               ArrayCopySemantics semantics),
              (final));
  MOCK_METHOD(Future<>, GetReadyFuture, (absl::Span<const ValueRef> values),
              (final));
  MOCK_METHOD(absl::StatusOr<tsl::RCReference<Tuple>>, MakeTuple,
              (absl::Span<ValueRef> values), (final));
  MOCK_METHOD(std::string_view, runtime_type, (), (const, final));
  MOCK_METHOD(std::string_view, platform_name, (), (const, final));
  MOCK_METHOD(std::string_view, platform_version, (), (const, final));
  MOCK_METHOD((const AttributeMap&), Attributes, (), (const, final));
  MOCK_METHOD(int, device_count, (), (const, final));
  MOCK_METHOD(PlatformId, platform_id, (), (const, final));
  MOCK_METHOD(int, addressable_device_count, (), (const, final));
  MOCK_METHOD(absl::Span<Device* const>, devices, (), (const, final));
  MOCK_METHOD(absl::Span<Device* const>, addressable_devices, (),
              (const, final));
  MOCK_METHOD(int, process_index, (), (const, final));
  MOCK_METHOD(absl::Span<Device* const>, GetAllDevices, (), (const, final));
  MOCK_METHOD(absl::StatusOr<DeviceAssignment>, GetDefaultDeviceAssignment,
              (int num_replicas, int num_partitions), (const, final));
  MOCK_METHOD(absl::StatusOr<Device*>, LookupDevice, (DeviceId device_id),
              (const, final));
  MOCK_METHOD(absl::StatusOr<Device*>, LookupAddressableDevice,
              (int local_hardware_id), (const, final));
  MOCK_METHOD(absl::StatusOr<DeviceListRef>, MakeDeviceList,
              (absl::Span<Device* const> devices), (const));
  MOCK_METHOD(Compiler*, GetDefaultCompiler, (), (final));
  MOCK_METHOD(absl::StatusOr<std::shared_ptr<Topology>>, GetTopologyForDevices,
              (const DeviceListRef& devices), (const, final));
  MOCK_METHOD(absl::StatusOr<std::shared_ptr<const zkx::PjRtLayout>>,
              GetDefaultPjRtLayout,
              (DType dtype, absl::Span<const int64_t> dims, Device* device,
               MemoryKind memory_kind),
              (const, final));
  MOCK_METHOD(absl::StatusOr<CustomLayoutRef>, GetDefaultLayout,
              (DType dtype, const Shape& shape, const ShardingRef& sharding),
              (const, final));
  MOCK_METHOD(tsl::RCReference<UserContext>, CreateUserContext, (), (final));
  // LINT.ThenChange(mock.cc:MockClientDelegation)

  Client* delegated() const { return delegated_.get(); }

  static char ID;

 private:
  const std::unique_ptr<Client> delegated_;
};

// compiler.h

class MockCompiler : public llvm::RTTIExtends<MockCompiler, Compiler> {
 public:
  MOCK_METHOD(absl::StatusOr<ExecutableRef>, Compile,
              (std::unique_ptr<Program> program, const Topology& topology,
               std::unique_ptr<CompileOptions> options),
              (final));
  MOCK_METHOD(absl::StatusOr<LoadedExecutableRef>, CompileAndLoad,
              (std::unique_ptr<Program> program,
               std::unique_ptr<CompileOptions> options),
              (final));
  MOCK_METHOD(absl::Status, IsExecutableVersionCompatible,
              (const ExecutableVersion& executable_version,
               const DeviceListRef& devices),
              (const, final));
  MOCK_METHOD(absl::StatusOr<LoadedExecutableRef>, DeserializeLoadedExecutable,
              (std::string_view serialized,
               std::unique_ptr<DeserializeExecutableOptions> options),
              (final));

  static char ID;
};

// device.h

class MockDevice : public Device {
 public:
  MockDevice() = default;
  explicit MockDevice(Device* delegated);

  // LINT.IfChange
  MOCK_METHOD(Client*, client, (), (const, final));
  MOCK_METHOD(bool, IsAddressable, (), (const, final));
  MOCK_METHOD(int, ProcessIndex, (), (const, final));
  MOCK_METHOD(DeviceId, Id, (), (const, final));
  MOCK_METHOD(std::string_view, Kind, (), (const, final));
  MOCK_METHOD((const AttributeMap&), Attributes, (), (const, final));
  MOCK_METHOD(absl::StatusOr<Memory*>, DefaultMemory, (), (const, final));
  MOCK_METHOD(absl::Span<Memory* const>, Memories, (), (const, final));
  // LINT.ThenChange(mock.cc:MockDeviceDelegation)

  Device* delegated() const { return delegated_; }

  std::string_view DebugString() const final { return "MockDevice"; }
  std::string_view ToString() const final { return "MockDevice"; }

 private:
  Device* const delegated_ = nullptr;
};

// device_list.h

class MockDeviceList : public DeviceList {
 public:
  MockDeviceList() = default;
  ~MockDeviceList() override = default;

  MOCK_METHOD(absl::Span<Device* const>, devices, (), (const final));
  MOCK_METHOD(DeviceList*, AddressableDeviceList, (), (const final));

  MOCK_METHOD(bool, EqualEqualOperator, (const DeviceList& other),
              (const final));
  bool operator==(const DeviceList& other) const override {
    return EqualEqualOperator(other);
  }
  MOCK_METHOD(uint64_t, hash, (), (const final));
  MOCK_METHOD(uint64_t, fingerprint, (), (const final));
  MOCK_METHOD(std::string, ToString, (), (const final));
};

// memory.h

class MockMemory : public Memory {
 public:
  MOCK_METHOD(MemoryId, Id, (), (const, final));
  MOCK_METHOD(absl::Span<Device* const>, Devices, (), (const, final));
  MOCK_METHOD(const MemoryKind&, Kind, (), (const, final));
  MOCK_METHOD(std::string_view, ToString, (), (const, final));

  std::string_view DebugString() const final { return "MockMemory"; }
};

// executable.h

class MockExecutable : public llvm::RTTIExtends<MockExecutable, Executable> {
 public:
  MOCK_METHOD(std::string_view, name, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::optional<std::string>>, Fingerprint, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<std::string>, Serialize, (), (const, final));
  MOCK_METHOD(int, num_devices, (), (const, final));
  MOCK_METHOD(int64_t, SizeOfGeneratedCodeInBytes, (), (const, final));
  MOCK_METHOD(absl::StatusOr<CompiledMemoryStats>, GetCompiledMemoryStats, (),
              (const, final));
  MOCK_METHOD(std::optional<std::vector<OpSharding>>, GetParameterShardings, (),
              (const, final));
  MOCK_METHOD(std::optional<std::vector<OpSharding>>, GetOutputShardings, (),
              (const, final));
  MOCK_METHOD(
      absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>,
      GetParameterLayouts, (), (const, final));
  MOCK_METHOD(
      absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>,
      GetOutputLayouts, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>,
              GetHloModules, (), (const, final));
  MOCK_METHOD(absl::StatusOr<AttributeMap>, GetCostAnalysis, (),
              (const, final));

  static char ID;
};

class MockLoadedExecutable
    : public llvm::RTTIExtends<MockLoadedExecutable, LoadedExecutable> {
 public:
  MockLoadedExecutable() {
    static absl::NoDestructor<DeviceListRef> kEmptyDeviceList(
        BasicDeviceList::Create({}));
    ON_CALL(*this, devices())
        .WillByDefault(testing::ReturnRef(*kEmptyDeviceList));
  }

  MOCK_METHOD(Client*, client, (), (const, final));
  MOCK_METHOD(std::string_view, name, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::optional<std::string>>, Fingerprint, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<ExecutableVersion>>,
              executable_version, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::string>, Serialize, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::string>, GetHumanReadableProgramText, (),
              (const, final));
  MOCK_METHOD(UserContextRef, user_context, (), (const, final));
  MOCK_METHOD(Future<>, GetReadyFuture, (), (const, final));
  MOCK_METHOD(int, num_devices, (), (const, final));
  MOCK_METHOD(int64_t, SizeOfGeneratedCodeInBytes, (), (const, final));
  MOCK_METHOD(absl::StatusOr<CompiledMemoryStats>, GetCompiledMemoryStats, (),
              (const, final));
  MOCK_METHOD(std::optional<std::vector<OpSharding>>, GetParameterShardings, (),
              (const, final));
  MOCK_METHOD(std::optional<std::vector<OpSharding>>, GetOutputShardings, (),
              (const, final));
  MOCK_METHOD(
      absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>,
      GetParameterLayouts, (), (const, final));
  MOCK_METHOD(absl::StatusOr<absl::Span<const int>>, GetDonatableInputIndices,
              (), (const, final));
  MOCK_METHOD(
      absl::StatusOr<std::vector<std::shared_ptr<const zkx::PjRtLayout>>>,
      GetOutputLayouts, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::vector<std::string_view>>>,
              GetOutputMemoryKinds, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>,
              GetHloModules, (), (const, final));
  MOCK_METHOD(absl::StatusOr<AttributeMap>, GetCostAnalysis, (),
              (const, final));
  MOCK_METHOD(absl::StatusOr<ExecuteResult>, Execute,
              (absl::Span<ArrayRef> args, const ExecuteOptions& options,
               std::optional<DeviceListRef> devices),
              (final));
  MOCK_METHOD(absl::Span<Device* const>, addressable_devices, (),
              (const, final));
  MOCK_METHOD(const DeviceListRef&, devices, (), (const, final));

  static char ID;
};

// host_callback.h

class MockHostCallback final
    : public llvm::RTTIExtends<MockHostCallback, HostCallback> {
 public:
  MOCK_METHOD(std::string, Serialize, (), (const, final));

  static char ID;
};

class MockLoadedHostCallback final
    : public llvm::RTTIExtends<MockLoadedHostCallback, LoadedHostCallback> {
 public:
  MOCK_METHOD(Client*, client, (), (const, final));
  MOCK_METHOD(absl::StatusOr<std::string>, Serialize, (), (const, final));

  static char ID;
};

// sharding.h

class MockSharding : public llvm::RTTIExtends<MockSharding, Sharding> {
 public:
  MockSharding()
      : llvm::RTTIExtends<MockSharding, Sharding>(
            BasicDeviceList::Create({}), MemoryKind(),
            /*is_fully_replicated=*/false) {}

  MockSharding(DeviceListRef devices, MemoryKind memory_kind,
               bool is_fully_replicated)
      : llvm::RTTIExtends<MockSharding, Sharding>(devices, memory_kind,
                                                  is_fully_replicated) {}

  MOCK_METHOD((absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>),
              Disassemble, (const Shape& shape), (const, final));
  MOCK_METHOD((absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>),
              Disassemble,
              (const Shape& shape,
               SingleDeviceShardSemantics single_device_shard_semantics),
              (const, final));
  MOCK_METHOD(
      (absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>),
      Disassemble, (const DynamicShape& dynamic_shape), (const final));
  MOCK_METHOD(
      (absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>),
      Disassemble,
      (const DynamicShape& dynamic_shape,
       SingleDeviceShardSemantics single_device_shard_semantics),
      (const final));
  MOCK_METHOD(absl::StatusOr<std::vector<IndexDomain>>, IndexDomains,
              (const Shape& shape), (const, final));
  MOCK_METHOD(absl::StatusOr<std::vector<IndexDomain>>, IndexDomains,
              (const Shape& shape,
               SingleDeviceShardSemantics single_device_shard_semantics),
              (const, final));
  MOCK_METHOD(absl::StatusOr<Shape>, GetShardShape, (const Shape& shape),
              (const, final));
  MOCK_METHOD(bool, HasSamePartitioning, (const Sharding& other),
              (const final));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Sharding>>, WithDeviceAssignment,
              (std::optional<DeviceListRef> devices,
               std::optional<MemoryKind> memory_kind),
              (const final));
  MOCK_METHOD(void, Hash, (absl::HashState), (const final));

  std::string DebugString() const final { return "MockSharding"; }

  static char ID;
};

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_IFRT_MOCK_H_
