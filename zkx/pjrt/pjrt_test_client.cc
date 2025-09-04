/* Copyright 2022 The OpenXLA Authors.

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

#include "zkx/pjrt/pjrt_test_client.h"

#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"

namespace zkx {
namespace {

class PjRtTestClientFactory {
 public:
  void Register(
      std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> factory) {
    absl::MutexLock lock(&mu_);
    CHECK(!factory_);
    factory_ = std::move(factory);
  }

  std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> Get() const {
    absl::MutexLock lock(&mu_);
    return factory_;
  }

 private:
  mutable absl::Mutex mu_;
  std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> factory_
      ABSL_GUARDED_BY(mu_);
};

PjRtTestClientFactory& GetPjRtTestClientFactory() {
  static auto* const factory = absl::IgnoreLeak(new PjRtTestClientFactory);
  return *factory;
}

}  // namespace

void RegisterPjRtTestClientFactory(
    std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> factory) {
  GetPjRtTestClientFactory().Register(std::move(factory));
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtTestClient() {
  return GetPjRtTestClientFactory().Get()();
}

}  // namespace zkx
