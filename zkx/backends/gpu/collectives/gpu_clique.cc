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

#include "zkx/backends/gpu/collectives/gpu_clique.h"

#include <stdint.h>

#include <utility>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"

namespace zkx::gpu {

GpuClique::GpuClique(
    GpuCliqueKey key, std::optional<CliqueIds> ids,
    absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators)
    : Clique(std::move(communicators)),
      key_(std::move(key)),
      ids_(std::move(ids)) {}

std::string GpuClique::DebugString() const {
  std::string out = absl::StrFormat(
      "key: %s; fingerprint(id): %d; size: %d; communicators: ",
      key_.ToString(), ids_.has_value() ? ids_->fingerprint() : 0,
      num_communicators());
  int32_t cnt = 0;
  ForEachComm([&](RankId rank, Communicator* comm) {
    if (cnt++) absl::StrAppend(&out, ", ");
    absl::StrAppendFormat(&out, "[rank=%d, comm=%p]", rank.value(), comm);
  });
  return out;
}

absl::Status GpuClique::HealthCheck() const {
  absl::Status health_check = absl::OkStatus();
  ForEachComm([&health_check](RankId rank, Communicator* comm) {
    if (auto s = comm->HealthCheck(); !s.ok()) {
      LOG(ERROR) << "GPU communicator error (rank " << rank << "): " << s;
      if (health_check.ok()) health_check = std::move(s);  // return first error
    }
  });
  return health_check;
}

std::string GpuClique::LockableName::ToString(const GpuClique& clique) {
  return absl::StrFormat("lockable clique %s", clique.key().ToString());
}

LockableGpuClique::LockableGpuClique(
    GpuCliqueKey clique_key, std::optional<CliqueIds> clique_ids,
    absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators)
    : Lockable(std::move(clique_key), clique_ids, std::move(communicators)) {}

absl::Status LockableGpuClique::HealthCheck() const {
  return value().HealthCheck();
}

std::string LockableGpuClique::DebugString() const {
  return absl::StrFormat("LockableGpuClique: %s", value().DebugString());
}

}  // namespace zkx::gpu
