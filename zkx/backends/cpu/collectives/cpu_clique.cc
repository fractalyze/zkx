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

#include "zkx/backends/cpu/collectives/cpu_clique.h"

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

namespace zkx::cpu {

absl::Status CpuClique::HealthCheck() const {
  absl::Status health_check = absl::OkStatus();
  ForEachComm([&health_check](RankId rank, Communicator* comm) {
    if (auto s = comm->HealthCheck(); !s.ok()) {
      LOG(ERROR) << "CPU communicator error (rank " << rank << "): " << s;
      if (health_check.ok()) health_check = std::move(s);  // return first error
    }
  });
  return health_check;
}

std::string CpuClique::DebugString() const {
  std::string out =
      absl::StrFormat("key: %s; size: %d; communicators: ", key_.ToString(),
                      num_communicators());
  int32_t cnt = 0;
  ForEachComm([&](RankId rank, Communicator* comm) {
    if (cnt++) absl::StrAppend(&out, ", ");
    absl::StrAppendFormat(&out, "[rank=%d, comm=%s]", rank.value(),
                          comm->ToString());
  });
  return out;
}

}  // namespace zkx::cpu
