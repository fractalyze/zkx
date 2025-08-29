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

#ifndef ZKX_BACKENDS_CPU_COLLECTIVES_CPU_COLLECTIVES_H_
#define ZKX_BACKENDS_CPU_COLLECTIVES_CPU_COLLECTIVES_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"

#include "zkx/core/collectives/collectives.h"
#include "zkx/service/collective_ops_utils.h"

namespace zkx::cpu {

// ZKX:CPU extension of the Collectives interface with CPU-specific APIs.
class CpuCollectives : public Collectives {
 public:
  // Returns the default collectives implementation for CPU backend.
  static CpuCollectives* Default();

  class Device : public Collectives::Device {
   public:
    Device() = default;
  };

  // Executor allows CPU collectives clients to pass additional information to
  // the collectives implementation.
  class Executor : public Communicator::Executor {
   public:
    Executor(RendezvousKey rendezvous_key, absl::Duration timeout)
        : rendezvous_key_(rendezvous_key), timeout_(timeout) {}

    const RendezvousKey& rendezvous_key() const { return rendezvous_key_; }
    const absl::Duration& timeout() const { return timeout_; }

   private:
    RendezvousKey rendezvous_key_;
    absl::Duration timeout_;
  };

  absl::StatusOr<CliqueId> CreateUniqueCliqueId() const final {
    return absl::UnimplementedError(
        "CPU collectives do not support clique ids");
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> SplitCommunicators(
      absl::Span<const Communicator* const> comms, int32_t color,
      absl::Span<const RankId> keys, const Config& config) final {
    return absl::UnimplementedError(
        "CPU collectives do not support communicator splitting");
  }

  // Tries to cast a Collectives::Device to a CpuCollectives::Device.
  static absl::StatusOr<const Device*> TryCast(
      const Collectives::Device* device);

  // Tries to cast a Communicator::Executor to a CpuCollectives::Executor.
  static absl::StatusOr<const Executor*> TryCast(
      const Communicator::Executor* executor);
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_COLLECTIVES_CPU_COLLECTIVES_H_
