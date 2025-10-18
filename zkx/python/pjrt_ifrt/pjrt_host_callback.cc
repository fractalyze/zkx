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

#include "zkx/python/pjrt_ifrt/pjrt_host_callback.h"

namespace zkx::ifrt {

char PjRtHostSendAndRecvLoadedHostCallback::ID = 0;

absl::StatusOr<std::string> PjRtHostSendAndRecvLoadedHostCallback::Serialize()
    const {
  return absl::UnimplementedError(
      "PjRtHostSendAndRecvLoadedHostCallback serialization is not supported");
}

char PjRtFfiLoadedHostCallback::ID = 0;

absl::StatusOr<std::string> PjRtFfiLoadedHostCallback::Serialize() const {
  return absl::UnimplementedError(
      "PjRtFfiLoadedHostCallback serialization is not supported");
}

}  // namespace zkx::ifrt
