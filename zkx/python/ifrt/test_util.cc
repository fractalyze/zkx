/* Copyright 2022 The OpenXLA Authors.
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

#include "zkx/python/ifrt/test_util.h"

#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

#include "zkx/python/ifrt/user_context_test_util.h"

namespace zkx::ifrt::test_util {
namespace {

class ClientFactory {
 public:
  void Register(
      std::function<absl::StatusOr<std::shared_ptr<Client>>()> factory) {
    absl::MutexLock lock(&mu_);
    CHECK(!factory_) << "Client factory has been already registered.";
    factory_ = std::move(factory);
  }

  std::function<absl::StatusOr<std::shared_ptr<Client>>()> Get() const {
    absl::MutexLock lock(&mu_);
    return factory_;
  }

 private:
  mutable absl::Mutex mu_;
  std::function<absl::StatusOr<std::shared_ptr<Client>>()> factory_
      ABSL_GUARDED_BY(mu_);
};

ClientFactory& GetGlobalClientFactory() {
  static auto* const factory = absl::IgnoreLeak(new ClientFactory);
  return *factory;
}

}  // namespace

void RegisterClientFactory(
    std::function<absl::StatusOr<std::shared_ptr<Client>>()> factory) {
  GetGlobalClientFactory().Register(std::move(factory));
}

absl::StatusOr<std::shared_ptr<Client>> GetClient() {
  auto factory = GetGlobalClientFactory().Get();
  CHECK(factory) << "Client factory has not been registered.";
  return factory();
}

void SetTestFilterIfNotUserSpecified(std::string_view custom_filter) {
  static constexpr std::string_view kDefaultTestFilter = "*";
#ifdef GTEST_FLAG_SET
  if (GTEST_FLAG_GET(filter) == kDefaultTestFilter) {
    GTEST_FLAG_SET(filter, custom_filter);
  }
#else
  if (testing::GTEST_FLAG(filter) == kDefaultTestFilter) {
    testing::GTEST_FLAG(filter) = custom_filter;
  }
#endif
}

absl::StatusOr<DeviceListRef> GetDevices(Client* client,
                                         absl::Span<const int> device_indices) {
  absl::InlinedVector<Device*, 1> devices;
  devices.reserve(device_indices.size());
  const absl::Span<Device* const> client_devices = client->devices();
  for (int device_index : device_indices) {
    if (device_index < 0 || device_index >= client_devices.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Out of range device index: ", device_index));
    }
    devices.push_back(client_devices[device_index]);
  }
  return client->MakeDeviceList(devices);
}

absl::StatusOr<DeviceListRef> GetAddressableDevices(
    Client* client, absl::Span<const int> device_indices) {
  absl::InlinedVector<Device*, 1> devices;
  devices.reserve(device_indices.size());
  const absl::Span<Device* const> client_devices =
      client->addressable_devices();
  for (int device_index : device_indices) {
    if (device_index < 0 || device_index >= client_devices.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Out of range device index: ", device_index));
    }
    devices.push_back(client_devices[device_index]);
  }
  return client->MakeDeviceList(std::move(devices));
}

UserContextRef MakeUserContext(uint64_t id) {
  return TestUserContext::Create(UserContextId(id));
}

}  // namespace zkx::ifrt::test_util
