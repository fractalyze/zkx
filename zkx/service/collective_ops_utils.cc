/* Copyright 2019 The OpenXLA Authors.

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

#include "zkx/service/collective_ops_utils.h"

namespace zkx {

std::string_view ReductionKindToString(ReductionKind reduction_kind) {
  switch (reduction_kind) {
    case ReductionKind::kSum:
      return "sum";
    case ReductionKind::kProduct:
      return "prod";
    case ReductionKind::kMin:
      return "min";
    case ReductionKind::kMax:
      return "max";
  }
}

std::string_view CollectiveOpGroupModeToString(
    CollectiveOpGroupMode group_mode) {
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica:
      return "kCrossReplica";
    case CollectiveOpGroupMode::kCrossPartition:
      return "kCrossPartition";
    case CollectiveOpGroupMode::kCrossReplicaAndPartition:
      return "kCrossReplicaAndPartition";
    case CollectiveOpGroupMode::kFlattenedID:
      return "kFlattenedID";
  }
}

// Returns the group formation mode implied by (a) whether the operation has
// channel_id and (b) if it has use_global_device_ids and if yes, its value.
absl::StatusOr<CollectiveOpGroupMode> GetCollectiveOpGroupMode(
    bool has_channel_id, std::optional<bool> use_global_device_ids) {
  if (!has_channel_id) {
    if (!use_global_device_ids.has_value() || !*use_global_device_ids) {
      return CollectiveOpGroupMode::kCrossReplica;
    } else {
      return absl::InvalidArgumentError(
          "Invalid combination of has_channel_id and use_global_device_ids");
    }
  } else {
    if (!use_global_device_ids.has_value()) {
      return CollectiveOpGroupMode::kCrossPartition;
    } else if (!*use_global_device_ids) {
      return CollectiveOpGroupMode::kCrossReplicaAndPartition;
    } else {
      return CollectiveOpGroupMode::kFlattenedID;
    }
  }
}

}  // namespace zkx
