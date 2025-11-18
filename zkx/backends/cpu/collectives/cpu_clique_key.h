/* Copyright 2025 The OpenXLA Authors.
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

#ifndef ZKX_BACKENDS_CPU_COLLECTIVES_CPU_CLIQUE_KEY_H_
#define ZKX_BACKENDS_CPU_COLLECTIVES_CPU_CLIQUE_KEY_H_

#include <string>

#include "absl/hash/hash.h"

#include "zkx/core/collectives/clique_key.h"

namespace zkx::cpu {

// Clique key for identifying a particular CPU collectives clique.
class CpuCliqueKey final : public CliqueKey {
 public:
  using CliqueKey::CliqueKey;

  bool IsSubsetOf(const CliqueKey& other) const final;
  std::string ToString() const final;

  bool operator==(const CpuCliqueKey& other) const {
    return devices() == other.devices();
  }
  bool operator<(const CpuCliqueKey& other) const {
    return devices() < other.devices();
  }
  bool operator>(const CpuCliqueKey& other) const {
    return devices() > other.devices();
  }

 private:
  void HashValue(absl::HashState state) const final;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_COLLECTIVES_CPU_CLIQUE_KEY_H_
