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

#include "zkx/core/collectives/clique_id.h"

#include "absl/crc/crc32c.h"

namespace zkx {

uint32_t CliqueId::fingerprint() const {
  std::string_view data_view(data_.data(), data_.size());
  return static_cast<uint32_t>(absl::ComputeCrc32c(data_view));
}

uint32_t CliqueIds::fingerprint() const {
  absl::crc32c_t crc(0);
  for (const auto& clique_id : ids_) {
    crc = absl::ExtendCrc32c(crc, std::string_view(clique_id.data().data(),
                                                   clique_id.data().size()));
  }
  return static_cast<uint32_t>(crc);
}

}  // namespace zkx
