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

#ifndef ZKX_PYTHON_PJRT_IFRT_ZKX_SHARDING_H_
#define ZKX_PYTHON_PJRT_IFRT_ZKX_SHARDING_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include "zkx/hlo/ir/hlo_sharding.h"
#include "zkx/python/ifrt/device_list.h"
#include "zkx/python/ifrt/index_domain.h"
#include "zkx/python/ifrt/memory.h"
#include "zkx/python/ifrt/shape.h"
#include "zkx/python/ifrt/sharding.h"

namespace zkx::ifrt {

// ZKX-compatible sharding types.
class ZkxCompatibleSharding
    : public llvm::RTTIExtends<ZkxCompatibleSharding, Sharding> {
 public:
  using llvm::RTTIExtends<ZkxCompatibleSharding, Sharding>::RTTIExtends;

  static char ID;
};

// ZKX `HloSharding` wrapper. `HloSharding` is the main sharding representation
// in ZKX. This class holds an `HloSharding` to be used with IFRT.
class HloSharding final
    : public llvm::RTTIExtends<HloSharding, ZkxCompatibleSharding> {
 public:
  // Creates an `HloSharding` wrapper. This bypasses consistency checks against
  // devices to optimize the common path of passing it to the user or to a
  // lower-level runtime. It is instead validated when the information in the
  // sharding is used within IFRT, e.g., in `Disassemble()`.
  static std::unique_ptr<HloSharding> Create(DeviceListRef devices,
                                             MemoryKind memory_kind,
                                             zkx::HloSharding zkx_hlo_sharding);

  // Returns the wrapped ZKX `HloSharding`.
  const zkx::HloSharding& zkx_hlo_sharding() const { return zkx_hlo_sharding_; }

  // Sharding implementation.

  ~HloSharding() override = default;

  absl::StatusOr<Shape> GetShardShape(const Shape& shape) const override;

  bool HasSamePartitioning(const Sharding& other) const override;

  absl::StatusOr<std::unique_ptr<Sharding>> WithDeviceAssignment(
      std::optional<DeviceListRef> devices,
      std::optional<MemoryKind> memory_kind) const override;

  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>> Disassemble(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape) const override;
  absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>> Disassemble(
      const DynamicShape& dynamic_shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape) const override;
  absl::StatusOr<std::vector<IndexDomain>> IndexDomains(
      const Shape& shape,
      SingleDeviceShardSemantics single_device_shard_semantics) const override;

  std::string DebugString() const override;

  static char ID;

 private:
  HloSharding(DeviceListRef devices, MemoryKind memory_kind,
              zkx::HloSharding zkx_hlo_sharding);

  void Hash(absl::HashState state) const override;

  zkx::HloSharding zkx_hlo_sharding_;

  // Cached hash. 0 indicates the hash needs to be computed and cached.
  // May be written multiple times with the same non-zero value.
  static constexpr uint64_t kUnsetHash = 0;
  mutable std::atomic<uint64_t> hash_ = kUnsetHash;
};

// Test only: returns `HloSharding::IndexDomains()`, using `zkx::HloSharding`
// APIs internally.
std::vector<IndexDomain> TEST_HloShardingIndexDomainsSlowPath(
    const HloSharding& sharding, const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics);

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_PJRT_IFRT_ZKX_SHARDING_H_
