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

#ifndef ZKX_CORE_COLLECTIVES_COLLECTIVES_REGISTRY_H_
#define ZKX_CORE_COLLECTIVES_COLLECTIVES_REGISTRY_H_

#include <stdint.h>

#include <memory>
#include <string_view>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "zkx/core/collectives/collectives.h"

namespace zkx {

// A registry of collective implementations registered with the current process.
class CollectivesRegistry {
 public:
  // Registers collective implementation for a given platform and name with a
  // given priority.
  //
  // The priority is used to determine which implementation is the default for
  // the given platform. Higher priority wins.
  //
  // Returns an error if the implementation is already registered.
  static absl::Status Register(std::string_view platform_name,
                               std::string_view name, int32_t priority,
                               std::unique_ptr<Collectives> collectives);

  // Returns the default collectives implementation for the given platform.
  static absl::StatusOr<Collectives*> Default(std::string_view platform_name);
};

}  // namespace zkx

#define ZKX_COLLECTIVES_REGISTER(PLATFORM, NAME, PRIORITY, IMPL) \
  ZKX_COLLECTIVES_REGISTER_(PLATFORM, NAME, PRIORITY, IMPL, __COUNTER__)
#define ZKX_COLLECTIVES_REGISTER_(PLATFORM, NAME, PRIORITY, IMPL, N) \
  ZKX_COLLECTIVES_REGISTER__(PLATFORM, NAME, PRIORITY, IMPL, N)
#define ZKX_COLLECTIVES_REGISTER__(PLATFORM, NAME, PRIORITY, IMPL, N)         \
  ABSL_ATTRIBUTE_UNUSED static const bool zkx_collectives_##N##_registered_ = \
      [] {                                                                    \
        absl::Status status = ::zkx::CollectivesRegistry::Register(           \
            PLATFORM, NAME, PRIORITY, IMPL);                                  \
        if (!status.ok()) {                                                   \
          LOG(ERROR) << "Failed to register XLA collectives: " << status;     \
        }                                                                     \
        return true;                                                          \
      }()

#endif  // ZKX_CORE_COLLECTIVES_COLLECTIVES_REGISTRY_H_
