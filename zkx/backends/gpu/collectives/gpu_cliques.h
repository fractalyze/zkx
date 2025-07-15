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

#ifndef ZKX_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUES_H_
#define ZKX_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUES_H_

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <memory>

#include "absl/container/btree_map.h"
#include "absl/status/statusor.h"

#include "zkx/backends/gpu/collectives/gpu_clique.h"
#include "zkx/backends/gpu/collectives/gpu_clique_key.h"

namespace zkx::gpu {

// A sorted container of acquired cliques. We keep cliques ordered by the key,
// so that all participants are guaranteed to iterate over the cliques in the
// same order, because otherwise we could get deadlocks when different
// participants try to split cliques in different orders.
class AcquiredCliquesMap
    : public absl::btree_map<GpuCliqueKey,
                             std::shared_ptr<LockableGpuClique::Lock>,
                             std::greater<GpuCliqueKey>> {};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUES_H_
