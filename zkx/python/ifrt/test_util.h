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

#ifndef ZKX_PYTHON_IFRT_TEST_UTIL_H_
#define ZKX_PYTHON_IFRT_TEST_UTIL_H_

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/python/ifrt/array.h"
#include "zkx/python/ifrt/client.h"
#include "zkx/python/ifrt/device.h"
#include "zkx/python/ifrt/device_list.h"
#include "zkx/python/ifrt/dtype.h"
#include "zkx/python/ifrt/shape.h"
#include "zkx/python/ifrt/sharding.h"
#include "zkx/python/ifrt/user_context.h"

namespace zkx::ifrt::test_util {

// Registers an IFRT client factory function. Must be called only once.
void RegisterClientFactory(
    std::function<absl::StatusOr<std::shared_ptr<Client>>()> factory);

// Gets a new IFRT client using the registered client factory.
absl::StatusOr<std::shared_ptr<Client>> GetClient();

// Set a default test filter if user doesn't provide one using --gtest_filter.
void SetTestFilterIfNotUserSpecified(std::string_view custom_filter);

// Asserts the content of an Array.
// This will blocking copy the data to host buffer.
template <typename ElementT>
void AssertPerShardData(
    ArrayRef actual, DType expected_dtype, Shape expected_per_shard_shape,
    absl::Span<const absl::Span<const ElementT>> expected_per_shard_data,
    DeviceListRef expected_device_list) {
  ASSERT_EQ(actual->dtype(), expected_dtype);
  EXPECT_THAT(GetDeviceIds(actual->sharding().devices()),
              testing::ElementsAreArray(GetDeviceIds(expected_device_list)));
  TF_ASSERT_OK_AND_ASSIGN(auto actual_per_shard_arrays,
                          actual->DisassembleIntoSingleDeviceArrays(
                              ArrayCopySemantics::kAlwaysCopy,
                              SingleDeviceShardSemantics::kAddressableShards));
  ASSERT_EQ(actual_per_shard_arrays.size(), expected_per_shard_data.size());
  for (int i = 0; i < actual_per_shard_arrays.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("Shard ", i));
    const ArrayRef& array = actual_per_shard_arrays[i];
    ASSERT_EQ(array->shape(), expected_per_shard_shape);
    std::vector<ElementT> actual_data(expected_per_shard_shape.num_elements());
    TF_ASSERT_OK(array
                     ->CopyToHostBuffer(actual_data.data(),
                                        /*byte_strides=*/std::nullopt,
                                        ArrayCopySemantics::kAlwaysCopy)
                     .Await());
    EXPECT_THAT(actual_data,
                testing::ElementsAreArray(expected_per_shard_data[i]));
  }
}

// Helper function that makes `DeviceList` containing devices at given
// indexes (not ids) within `client.devices()`.
absl::StatusOr<DeviceListRef> GetDevices(Client* client,
                                         absl::Span<const int> device_indices);

// Helper function that makes `DeviceList` containing devices at given
// indexes (not ids) within `client.addressable_devices()`.
absl::StatusOr<DeviceListRef> GetAddressableDevices(
    Client* client, absl::Span<const int> device_indices);

// Returns a new `UserContext` for testing. The created `UserContext` has an
// ID equal to `id`.
UserContextRef MakeUserContext(uint64_t id);
}  // namespace zkx::ifrt::test_util

#endif  // ZKX_PYTHON_IFRT_TEST_UTIL_H_
