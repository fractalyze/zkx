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

#ifndef ZKX_SERVICE_SHARDING_CONFIG_H_
#define ZKX_SERVICE_SHARDING_CONFIG_H_

#include <optional>
#include <vector>

#include "zkx/hlo/ir/hlo_sharding.h"
#include "zkx/zkx.pb.h"

namespace zkx {

// Node's sharding config, where sharding represents the sharding for a
// non-tuple, and nodes[i] represents the sharding for the i-th tuple element.
struct NodeShardingConfig {
  std::optional<HloSharding> sharding;
  std::vector<NodeShardingConfig> nodes;
  bool operator==(const NodeShardingConfig& other) const {
    return sharding == other.sharding && nodes == other.nodes;
  }
};

// Program's sharding configuration.
struct ShardingConfig {
  std::vector<NodeShardingConfig> nodes;
  bool operator==(const ShardingConfig& other) const {
    return nodes == other.nodes;
  }
  static ShardingConfig FromProto(const ShardingConfigProto& proto);
  static ShardingConfigProto ToProto(const ShardingConfig& config);
};

}  // namespace zkx

#endif  // ZKX_SERVICE_SHARDING_CONFIG_H_
