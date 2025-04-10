/* Copyright 2024 The OpenXLA Authors.

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

#include "zkx/hlo/ir/collective_device_list.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "zkx/base/logging.h"

namespace zkx {

std::string ReplicaGroupsToString(
    absl::Span<const ReplicaGroup> replica_groups) {
  std::vector<std::string> replica_group_str;
  replica_group_str.reserve(replica_groups.size());
  for (const ReplicaGroup& group : replica_groups) {
    replica_group_str.push_back(
        absl::StrCat("{", absl::StrJoin(group.replica_ids(), ","), "}"));
  }
  return absl::StrCat("{", absl::StrJoin(replica_group_str, ","), "}");
}

void CollectiveDeviceList::MaybeMaterializeFullReplicaGroupList() const {
  if (replica_groups_ != nullptr && !replica_groups_->empty()) {
    VLOG(10) << "Replica group list already materialized.";
    return;
  }
  if (!iota_replica_group_list_.has_value()) {
    VLOG(1) << "Replica group list not materialized because iota replica group "
               "list is not present.";
    return;
  }
  VLOG(10) << "Materializing full replica group list";

  replica_groups_ = std::make_shared<std::vector<ReplicaGroup>>();
  const int64_t num_replica_groups =
      iota_replica_group_list_->num_replica_groups();
  replica_groups_->reserve(num_replica_groups);

  Array<int64_t> array = iota_replica_group_list_->ToArray();
  // Iota replica group list array must only have 2 dimensions.
  DCHECK_EQ(array.num_dimensions(), 2);
  const int64_t num_devices_per_group =
      iota_replica_group_list_->num_devices_per_group();
  DCHECK_EQ(array.end() - array.begin(),
            num_devices_per_group * num_replica_groups);
  for (auto it = array.begin(); it != array.end();
       it += num_devices_per_group) {
    auto& group = replica_groups_->emplace_back();
    *group.mutable_replica_ids() = {it, it + num_devices_per_group};
  }
}

std::string CollectiveDeviceList::ToString(
    bool print_full_replica_group_list) const {
  if (iota_replica_group_list_.has_value() && !print_full_replica_group_list) {
    return iota_replica_group_list_->ToString();
  }

  return ReplicaGroupsToString(replica_groups());
}

}  // namespace zkx
