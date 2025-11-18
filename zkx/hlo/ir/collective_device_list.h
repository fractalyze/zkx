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

#ifndef ZKX_HLO_IR_COLLECTIVE_DEVICE_LIST_H_
#define ZKX_HLO_IR_COLLECTIVE_DEVICE_LIST_H_

#include <stdint.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"

#include "zkx/array.h"
#include "zkx/hlo/ir/tile_assignment.h"
#include "zkx/service/hlo.pb.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

std::string ReplicaGroupsToString(
    absl::Span<const ReplicaGroup> replica_groups);

// Represents a list of replica groups (a list of list of devices) with
// reshaping and transposing an iota array (iota tile assignment). Can be used
// to represent certain common patterns of device lists in a compact, scalable
// format.
class IotaReplicaGroupList {
 public:
  explicit IotaReplicaGroupList(int64_t num_replica_groups,
                                int64_t num_devices_per_group)
      : iota_tile_assignment_(IotaTileAssignment::Create(
            {num_replica_groups, num_devices_per_group})),
        num_replica_groups_(num_replica_groups),
        num_devices_per_group_(num_devices_per_group) {}

  explicit IotaReplicaGroupList(int64_t num_replica_groups,
                                int64_t num_devices_per_group,
                                absl::Span<const int64_t> reshape_dims,
                                absl::Span<const int> transpose_perm)
      : iota_tile_assignment_(IotaTileAssignment::Create(
            {num_replica_groups, num_devices_per_group}, reshape_dims,
            transpose_perm)),
        num_replica_groups_(num_replica_groups),
        num_devices_per_group_(num_devices_per_group) {}

  int64_t num_replica_groups() const {
    DCHECK_GE(num_replica_groups_, 0);
    return num_replica_groups_;
  }
  int64_t num_devices_per_group() const {
    DCHECK_GE(num_devices_per_group_, 0);
    return num_devices_per_group_;
  }
  absl::Span<const int64_t> reshape_dims() const {
    return iota_tile_assignment_.reshape_dims();
  }
  absl::Span<const int> transpose_perm() const {
    return iota_tile_assignment_.transpose_perm();
  }
  Array<int64_t> ToArray() const { return iota_tile_assignment_.ToArray(); }

  std::string ToString() const { return iota_tile_assignment_.ToString(); }

  IotaReplicaGroupListProto ToProto() const;

  static IotaReplicaGroupList FromProto(const IotaReplicaGroupListProto& proto);

 private:
  IotaTileAssignment iota_tile_assignment_;
  int64_t num_replica_groups_ = -1;
  int64_t num_devices_per_group_ = -1;
};

// Represents a series of devices participating in a collective operation
// (all-gather, all-reduce, etc.). While this directly translates to a list of
// replica groups, it may be used to represent these lists in compact forms.
class CollectiveDeviceList {
 public:
  explicit CollectiveDeviceList() = default;

  explicit CollectiveDeviceList(absl::Span<const ReplicaGroup> replica_groups)
      : replica_groups_(std::make_shared<std::vector<ReplicaGroup>>(
            replica_groups.begin(), replica_groups.end())) {}

  explicit CollectiveDeviceList(
      absl::Span<const std::vector<int64_t>> replica_groups)
      : replica_groups_(ToReplicaGroupVector(replica_groups)) {}

  // Replica groups are materialized lazily upon first access.
  explicit CollectiveDeviceList(
      const IotaReplicaGroupList& iota_replica_group_list)
      : iota_replica_group_list_(iota_replica_group_list) {}

  const std::vector<ReplicaGroup>& replica_groups() const {
    MaybeMaterializeFullReplicaGroupList();
    return *replica_groups_;
  }

  const std::optional<IotaReplicaGroupList>& iota_replica_group_list() const {
    return iota_replica_group_list_;
  }

  std::string ToString(bool print_full_replica_group_list = false) const;

  CollectiveDeviceListProto ToProto() const;

  static CollectiveDeviceList FromProto(const CollectiveDeviceListProto& proto);

  static CollectiveDeviceList FromProto(const HloInstructionProto& proto);

 private:
  // Construct collective device list from protobuf replica group start and end
  // iterators.
  CollectiveDeviceList(
      google::protobuf::RepeatedPtrField<ReplicaGroup>::const_iterator start,
      google::protobuf::RepeatedPtrField<ReplicaGroup>::const_iterator end)
      : replica_groups_(
            std::make_shared<std::vector<ReplicaGroup>>(start, end)) {}

  static std::shared_ptr<std::vector<ReplicaGroup>> ToReplicaGroupVector(
      absl::Span<const std::vector<int64_t>> replica_groups) {
    std::shared_ptr<std::vector<ReplicaGroup>> result =
        std::make_shared<std::vector<ReplicaGroup>>();
    result->reserve(replica_groups.size());
    for (const std::vector<int64_t>& g : replica_groups) {
      auto& group = result->emplace_back();
      group.mutable_replica_ids()->Add(g.begin(), g.end());
    }
    return result;
  }

  // Load replica groups from iota tile assignment if not already done so.
  void MaybeMaterializeFullReplicaGroupList() const;

  std::optional<IotaReplicaGroupList> iota_replica_group_list_;
  // shared_ptr for fast copy.
  mutable std::shared_ptr<std::vector<ReplicaGroup>> replica_groups_ =
      std::make_shared<std::vector<ReplicaGroup>>();
};

}  // namespace zkx

#endif  // ZKX_HLO_IR_COLLECTIVE_DEVICE_LIST_H_
