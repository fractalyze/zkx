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

#include "zkx/core/collectives/clique_key.h"

#include "absl/algorithm/container.h"

namespace zkx {

std::optional<RankId> CliqueKey::rank(GlobalDeviceId id) const {
  if (auto it = absl::c_find(devices_, id); it != devices_.end()) {
    return RankId(it - devices_.begin());
  }
  return std::nullopt;
}
}  // namespace zkx
