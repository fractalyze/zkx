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

#include "zkx/python/pjrt_ifrt/zkx_sharding.h"

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/python/ifrt/device.h"
#include "zkx/python/ifrt/index.h"
#include "zkx/shape_util.h"
#include "zkx/util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx::ifrt {

char ZkxCompatibleSharding::ID = 0;
char HloSharding::ID = 0;

namespace {

// Generates IndexDomains for an HloSharding, using ZKX HloSharding APIs.
// Note that this is O(N^2) where N is the number of devices (shards).
std::vector<IndexDomain> IndexDomainsSlowPath(
    const zkx::HloSharding& hlo_sharding, const DeviceListRef& devices,
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  // Only shape dimensions are used.
  auto zkx_shape = ShapeUtil::MakeShapeWithDescendingLayout(PrimitiveType::S32,
                                                            shape.dims());
  if (devices->size() > 8) {
    LOG_FIRST_N(WARNING, 1)
        << "Taking a slow path for HloSharding::IndexDomains(). This will not "
           "scale for a large number of devices.";
  }

  std::vector<IndexDomain> result;
  result.reserve(devices->size());

  Index::Elements origin(shape.dims().size());
  Shape::Dimensions shard_shape(shape.dims().size());
  const absl::Span<Device* const> device_ptrs = devices->devices();
  for (int device_idx = 0; device_idx < device_ptrs.size(); ++device_idx) {
    if (single_device_shard_semantics ==
            SingleDeviceShardSemantics::kAllShards ||
        device_ptrs[device_idx]->IsAddressable()) {
      auto tile_offset =
          hlo_sharding.TileOffsetForDevice(zkx_shape, device_idx);
      auto tile_limit = hlo_sharding.TileLimitForDevice(zkx_shape, device_idx);
      for (int i = 0; i < shape.dims().size(); ++i) {
        origin[i] = tile_offset[i];
        shard_shape[i] = tile_limit[i] - tile_offset[i];
      }
      result.push_back(IndexDomain(Index(origin), Shape(shard_shape)));
    }
  }
  return result;
}

// Returns a canonicalized memory kind for the given devices.
// REQUIRES: !devices->devices().empty()
MemoryKind CanonicalizeMemoryKindWithDevices(const MemoryKind& memory_kind,
                                             const DeviceListRef& devices) {
  CHECK_NE(devices, nullptr);
  CHECK(!devices->devices().empty());
  return CanonicalizeMemoryKind(memory_kind, devices->devices().front());
}

}  // namespace

// static
std::unique_ptr<HloSharding> HloSharding::Create(
    DeviceListRef devices, MemoryKind memory_kind,
    zkx::HloSharding zkx_hlo_sharding) {
  memory_kind = CanonicalizeMemoryKindWithDevices(memory_kind, devices);
  return std::unique_ptr<HloSharding>(new HloSharding(
      std::move(devices), memory_kind, std::move(zkx_hlo_sharding)));
}

HloSharding::HloSharding(DeviceListRef devices, MemoryKind memory_kind,
                         zkx::HloSharding zkx_hlo_sharding)
    : llvm::RTTIExtends<HloSharding, ZkxCompatibleSharding>(
          std::move(devices), memory_kind,
          // Computed in the constructor because it needs to access `devices` or
          // `devices_`; this access would be unsafe unless `device` is not
          // moved.
          /*is_fully_replicated=*/false),
      zkx_hlo_sharding_(std::move(zkx_hlo_sharding)) {
  is_fully_replicated_ =
      zkx_hlo_sharding_.IsReplicated() ||
      ((zkx_hlo_sharding_.IsTiled() || zkx_hlo_sharding_.IsTileMaximal()) &&
       devices_->size() == 1);
}

absl::StatusOr<Shape> HloSharding::GetShardShape(const Shape& shape) const {
  if (zkx_hlo_sharding_.IsTileMaximal() || zkx_hlo_sharding_.IsManual() ||
      zkx_hlo_sharding_.IsUnknown()) {
    return shape;
  }
  // TODO(chokobole): Uncomment this. Dependency: HloSharding::IsUnreduced()
  // if (zkx_hlo_sharding_.IsUnreduced()) {
  //   return shape;
  // }
  if (zkx_hlo_sharding_.TotalNumTiles() != devices_->size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("sharding's tile count and device count does not "
                        "match: %d vs. %d; shape=%s, sharding=%s",
                        zkx_hlo_sharding_.TotalNumTiles(), devices_->size(),
                        shape.DebugString(), zkx_hlo_sharding_.ToString()));
  }
  if (shape.dims().size() != zkx_hlo_sharding_.TiledDataRank()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Numbers of dimensions don't match. From Shape %d vs from "
        "HloSharding %d",
        shape.dims().size(), zkx_hlo_sharding_.TiledDataRank()));
  }
  const absl::Span<const int64_t> tile_assignment_dims =
      zkx_hlo_sharding_.tile_assignment().dimensions();
  Shape::Dimensions tile_shape;
  tile_shape.reserve(shape.dims().size());
  for (int64_t i = 0; i < shape.dims().size(); ++i) {
    tile_shape.push_back(CeilOfRatio(shape.dims()[i], tile_assignment_dims[i]));
  }
  return Shape(std::move(tile_shape));
}

bool HloSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  if (devices()->size() != other.devices()->size()) {
    return false;
  }
  const auto* other_hlo_sharding = llvm::dyn_cast<HloSharding>(&other);
  if (!other_hlo_sharding) {
    return false;
  }
  return zkx_hlo_sharding_ == other_hlo_sharding->zkx_hlo_sharding_;
}

absl::StatusOr<std::unique_ptr<Sharding>> HloSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "HloSharding should have the same number of devices as the current "
        "sharding, but was asked to have %d devices",
        (*devices)->size()));
  }
  return Create(devices.value_or(devices_), memory_kind.value_or(memory_kind_),
                zkx_hlo_sharding_);
}
absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
HloSharding::Disassemble(const Shape& shape) const {
  return Disassemble(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
HloSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  bool is_even_sharding = false;
  if (zkx_hlo_sharding_.IsReplicated() || zkx_hlo_sharding_.IsTileMaximal()) {
    is_even_sharding = true;
  }
  // TODO(chokobole): Uncomment this. Dependency: HloSharding::IsUnreduced()
  // else if (zkx_hlo_sharding_.IsUnreduced()) {
  //   is_even_sharding = true;
  // }
  // NOLINTNEXTLINE(readability/braces)
  else if (zkx_hlo_sharding_.IsTiled()) {
    const int64_t tiled_data_rank = zkx_hlo_sharding_.TiledDataRank();
    if (shape.dims().size() != tiled_data_rank) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "shape must have %d dimensions, but has %d dimensions: "
          "shape=%s, sharding=%s",
          tiled_data_rank, shape.dims().size(), shape.DebugString(),
          zkx_hlo_sharding_.ToString()));
    }

    is_even_sharding = true;
    for (int i = 0; i < tiled_data_rank; ++i) {
      if (shape.dims()[i] % zkx_hlo_sharding_.tile_assignment().dim(i) != 0) {
        is_even_sharding = false;
        break;
      }
    }
  } else if (zkx_hlo_sharding_.IsManual()) {
    // By convention, MANUAL sharding has the same global/shard shapes.
    is_even_sharding = true;
  }

  const absl::Span<Device* const> devices = devices_->devices();
  if (is_even_sharding) {
    // Fast path for even sharding.
    TF_ASSIGN_OR_RETURN(zkx::ifrt::Shape shard_shape, GetShardShape(shape));
    std::vector<std::pair<Shape, ShardingRef>> result;
    if (single_device_shard_semantics ==
        SingleDeviceShardSemantics::kAllShards) {
      result.reserve(devices_->size());
    } else {
      result.reserve(devices_->AddressableDeviceList()->size());
    }
    for (int i = 0; i < devices_->size(); ++i) {
      if (single_device_shard_semantics ==
              SingleDeviceShardSemantics::kAllShards ||
          devices[i]->IsAddressable()) {
        result.push_back({
            shard_shape,
            SingleDeviceSharding::Create(devices[i], memory_kind_),
        });
      }
    }
    return result;
  }
  // Slow path that uses `IndexDomains()` to handle uneven sharding.
  TF_ASSIGN_OR_RETURN(std::vector<IndexDomain> index_domains,
                      IndexDomains(shape, single_device_shard_semantics));
  std::vector<std::pair<Shape, ShardingRef>> result;
  result.reserve(index_domains.size());

  const absl::Span<Device* const> disassembled_devices =
      (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards)
          ? devices_->devices()
          : devices_->AddressableDeviceList()->devices();
  CHECK_EQ(index_domains.size(), disassembled_devices.size());

  for (int i = 0; i < index_domains.size(); ++i) {
    result.push_back(
        {index_domains[i].shape(),
         SingleDeviceSharding::Create(disassembled_devices[i], memory_kind_)});
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
HloSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  return Disassemble(dynamic_shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
HloSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  return absl::InvalidArgumentError(absl::StrFormat(
      "HloSharding can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %s",
      dynamic_shape.DebugString()));
}

absl::StatusOr<std::vector<IndexDomain>> HloSharding::IndexDomains(
    const Shape& shape) const {
  return IndexDomains(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<IndexDomain>> HloSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  std::vector<IndexDomain> result;
  const int num_devices = devices_->size();

  if (zkx_hlo_sharding_.IsManual()) {
    return absl::InvalidArgumentError(
        "Manual sharding does not support IndexDomains");
  }
  if (zkx_hlo_sharding_.IsReplicated() || zkx_hlo_sharding_.IsTileMaximal()) {
    // Fast path for a fully replicated or maximal sharding.
    IndexDomain element(shape);
    if (single_device_shard_semantics ==
        SingleDeviceShardSemantics::kAllShards) {
      result.resize(/*count=*/num_devices, /*value=*/element);
    } else {
      result.resize(/*count=*/devices_->AddressableDeviceList()->size(),
                    /*value=*/element);
    }
    return result;
  }
  if (!zkx_hlo_sharding_.IsTiled()) {
    return IndexDomainsSlowPath(zkx_hlo_sharding_, devices_, shape,
                                single_device_shard_semantics);
  }
  for (const OpSharding::Type subgroup_type :
       zkx_hlo_sharding_.subgroup_types()) {
    if (subgroup_type != OpSharding::REPLICATED) {
      return IndexDomainsSlowPath(zkx_hlo_sharding_, devices_, shape,
                                  single_device_shard_semantics);
    }
  }
  if (zkx_hlo_sharding_.tile_assignment().num_elements() != num_devices) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "sharding's tile_assignment_devices and device count does not "
        "match: %d vs. %d; shape=%s, sharding=%s",
        zkx_hlo_sharding_.tile_assignment().num_elements(), num_devices,
        shape.DebugString(), DebugString()));
  }

  const int64_t tiled_data_rank = zkx_hlo_sharding_.TiledDataRank();
  if (shape.dims().size() != tiled_data_rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("shape must have %d dimensions, but has %d dimensions: "
                        "shape=%s, sharding=%s",
                        tiled_data_rank, shape.dims().size(),
                        shape.DebugString(), zkx_hlo_sharding_.ToString()));
  }

  TF_ASSIGN_OR_RETURN(Shape tile_shape, GetShardShape(shape));

  const absl::Span<const int64_t> shape_dims = shape.dims();
  std::vector<std::optional<IndexDomain>> all(num_devices);
  // TODO(chokobole): Implement this. Dependency: HloSharding::EachTile()
  return absl::UnimplementedError(
      "HloSharding::IndexDomains() is not implemented");
}

std::string HloSharding::DebugString() const {
  return absl::StrFormat("HloSharding(memory_kind: %v, hlo_sharding: %s)",
                         memory_kind_, zkx_hlo_sharding_.ToString());
}

void HloSharding::Hash(absl::HashState state) const {
  uint64_t hash = hash_.load(std::memory_order_relaxed);
  if (hash == kUnsetHash) {
    hash = absl::HashOf(devices_, memory_kind_, zkx_hlo_sharding_);
    if (ABSL_PREDICT_FALSE(hash == kUnsetHash)) {
      ++hash;
    }
    hash_.store(hash, std::memory_order_relaxed);
  }
  absl::HashState::combine(std::move(state), hash);
}

std::vector<IndexDomain> TEST_HloShardingIndexDomainsSlowPath(
    const HloSharding& hlo_sharding, const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  return IndexDomainsSlowPath(hlo_sharding.zkx_hlo_sharding(),
                              hlo_sharding.devices(), shape,
                              single_device_shard_semantics);
}

}  // namespace zkx::ifrt
