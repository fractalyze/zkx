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

#ifndef ZKX_PYTHON_PJRT_IFRT_PJRT_HOST_CALLBACK_H_
#define ZKX_PYTHON_PJRT_IFRT_PJRT_HOST_CALLBACK_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include "zkx/pjrt/host_callback.h"
#include "zkx/python/ifrt/client.h"
#include "zkx/python/ifrt/host_callback.h"

namespace zkx::ifrt {

// Wrapper of a PjRt `zkx::HostCallback` that uses ZKX host send and recv. This
// object is expected to be passed to the compiler when creating
// `zkx::ifrt::PjRtLoadedExecutable`.
//
// `PjRtHostSendAndRecvLoadedHostCallback` does not support serialization by
// default, but it may be implemented by subclassing it.
//
// TODO(hyeontaek): Update the comment (compiler to client) after splitting
// compilation and loading.
class PjRtHostSendAndRecvLoadedHostCallback
    : public llvm::RTTIExtends<PjRtHostSendAndRecvLoadedHostCallback,
                               LoadedHostCallback> {
 public:
  PjRtHostSendAndRecvLoadedHostCallback(
      Client* client, std::unique_ptr<zkx::HostCallback> host_callback)
      : client_(client), host_callback_(std::move(host_callback)) {}

  const zkx::HostCallback& host_callback() const { return *host_callback_; }

  // LoadedHostCallback implementation.

  ~PjRtHostSendAndRecvLoadedHostCallback() override = default;

  Client* client() const override { return client_; }

  absl::StatusOr<std::string> Serialize() const override;

  static char ID;

 private:
  Client* client_;  // not owned
  std::unique_ptr<zkx::HostCallback> host_callback_;
};

// Wrapper of an opaque callable that is loaded into FFI's ExecutionContext
// during execution.
class PjRtFfiLoadedHostCallback
    : public llvm::RTTIExtends<PjRtFfiLoadedHostCallback, LoadedHostCallback> {
 public:
  explicit PjRtFfiLoadedHostCallback(Client* client, void* callable)
      : client_(client), callable_(callable) {}

  ~PjRtFfiLoadedHostCallback() override = default;

  Client* client() const override { return client_; }

  void* callable() const { return callable_; }

  absl::StatusOr<std::string> Serialize() const override;

  static char ID;

 private:
  Client* client_;  // not owned
  void* callable_;  // not owned
};

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_PJRT_IFRT_PJRT_HOST_CALLBACK_H_
