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

#ifndef ZKX_TOOLS_MULTIHOST_HLO_RUNNER_CREATE_CLIENT_H_
#define ZKX_TOOLS_MULTIHOST_HLO_RUNNER_CREATE_CLIENT_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/time/time.h"

#include "zkx/pjrt/distributed/client.h"
#include "zkx/pjrt/distributed/key_value_store_interface.h"
#include "zkx/pjrt/distributed/service.h"
#include "zkx/pjrt/pjrt_client.h"

namespace zkx {

struct PjRtEnvironment {
  // Sequence matters here, client should be destroyed before service.
  std::unique_ptr<DistributedRuntimeService> service;
  std::unique_ptr<PjRtClient> client;
  std::shared_ptr<KeyValueStoreInterface> kv_store;
  std::shared_ptr<DistributedRuntimeClient> distributed_client;
};

// Creates an environment with a PjRtClient for host CPU.
absl::StatusOr<PjRtEnvironment> GetPjRtEnvironmentForHostCpu();

// Creates a PjRtClient which can run HLOs on Host CPU.
absl::StatusOr<std::unique_ptr<PjRtClient>> CreateHostClient();

}  // namespace zkx

#endif  // ZKX_TOOLS_MULTIHOST_HLO_RUNNER_CREATE_CLIENT_H_
