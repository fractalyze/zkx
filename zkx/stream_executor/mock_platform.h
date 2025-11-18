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

#ifndef ZKX_STREAM_EXECUTOR_MOCK_PLATFORM_H_
#define ZKX_STREAM_EXECUTOR_MOCK_PLATFORM_H_

#include "gmock/gmock.h"

#include "zkx/stream_executor/platform.h"

namespace stream_executor {

// Implements the Platform interface for testing.
class MockPlatform : public Platform {
 public:
  MockPlatform() = default;
  MOCK_METHOD(Id, id, (), (const, override));
  MOCK_METHOD(const std::string&, Name, (), (const, override));
  MOCK_METHOD(int, VisibleDeviceCount, (), (const, override));
  MOCK_METHOD(bool, Initialized, (), (const, override));
  MOCK_METHOD(absl::Status, Initialize, (), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<DeviceDescription>>,
              DescriptionForDevice, (int ordinal), (const, override));
  MOCK_METHOD(absl::StatusOr<StreamExecutor*>, ExecutorForDevice, (int ordinal),
              (override));
  MOCK_METHOD(absl::StatusOr<StreamExecutor*>, FindExisting, (int ordinal),
              (override));
};

}  // namespace stream_executor

#endif  // ZKX_STREAM_EXECUTOR_MOCK_PLATFORM_H_
