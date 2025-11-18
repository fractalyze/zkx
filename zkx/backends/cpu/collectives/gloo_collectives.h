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

#ifndef ZKX_BACKENDS_CPU_COLLECTIVES_GLOO_COLLECTIVES_H_
#define ZKX_BACKENDS_CPU_COLLECTIVES_GLOO_COLLECTIVES_H_

#include <utility>

#include "gloo/rendezvous/store.h"
#include "gloo/transport/device.h"

#include "zkx/backends/cpu/collectives/cpu_collectives.h"

namespace zkx::cpu {

class GlooCollectives : public CpuCollectives {
 public:
  GlooCollectives(std::unique_ptr<gloo::rendezvous::Store> store,
                  std::shared_ptr<gloo::transport::Device> device)
      : store_(std::move(store)), device_(std::move(device)) {}
  ~GlooCollectives() override = default;

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey& clique_key,
                      const std::optional<CliqueIds>& clique_ids,
                      absl::Span<const DeviceRank> ranks,
                      const Config& config) final;

 private:
  std::unique_ptr<gloo::rendezvous::Store> store_;
  std::shared_ptr<gloo::transport::Device> device_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_COLLECTIVES_GLOO_COLLECTIVES_H_
