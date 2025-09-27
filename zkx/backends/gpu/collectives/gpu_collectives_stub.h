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

#ifndef ZKX_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_STUB_H_
#define ZKX_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_STUB_H_

#include "zkx/backends/gpu/collectives/gpu_collectives.h"

namespace zkx::gpu {

// A stub for GPU collectives when ZKX:GPU compiled without collectives support.
class GpuCollectivesStub : public GpuCollectives {
 public:
  bool IsImplemented() const final { return false; }
  bool IsGlobalConfig() const final { return false; }

  absl::StatusOr<CliqueId> CreateUniqueCliqueId() const final {
    return UnimplementedError();
  }

  absl::StatusOr<const CliqueIdCallback*> GetCliqueIdCallback(
      const CliqueIdCallback*, bool) final {
    return UnimplementedError();
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
  CreateCommunicators(const CliqueKey&, const std::optional<CliqueIds>&,
                      absl::Span<const DeviceRank>,
                      const Collectives::Config&) final {
    return UnimplementedError();
  }

  absl::StatusOr<std::vector<std::unique_ptr<Communicator>>> SplitCommunicators(
      absl::Span<const Communicator* const>, int32_t, absl::Span<const RankId>,
      const Collectives::Config&) final {
    return UnimplementedError();
  }

  absl::Status GroupStart() final { return UnimplementedError(); }
  absl::Status GroupEnd() final { return UnimplementedError(); }

 protected:
  static absl::Status UnimplementedError() {
    return absl::UnimplementedError(
        "ZKX compiled without GPU collectives support");
  }
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_STUB_H_
