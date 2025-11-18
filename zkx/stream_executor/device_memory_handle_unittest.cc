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
#include "zkx/stream_executor/device_memory_handle.h"

#include <utility>

#include "gtest/gtest.h"

#include "zkx/stream_executor/mock_stream_executor.h"

namespace stream_executor {

TEST(DeviceMemoryHandle, NullMemoryNoDeallocate) {
  DeviceMemoryBase null_memory;
  MockStreamExecutor executor;
  EXPECT_CALL(executor, Deallocate).Times(0);
  {
    DeviceMemoryHandle releaser(&executor, null_memory);
  }
}

TEST(DeviceMemoryHandle, Deallocates) {
  MockStreamExecutor executor;
  DeviceMemoryBase memory(&executor, sizeof(executor));
  EXPECT_CALL(executor, Deallocate).Times(1);
  {
    DeviceMemoryHandle releaser(&executor, memory);
  }
}

TEST(DeviceMemoryHandle, MoveDeallocatesOnce) {
  MockStreamExecutor executor;
  DeviceMemoryBase memory(&executor, sizeof(executor));
  EXPECT_CALL(executor, Deallocate).Times(1);
  {
    DeviceMemoryHandle releaser(&executor, memory);
    DeviceMemoryHandle releaser_moved(std::move(releaser));
  }
}

TEST(DeviceMemoryHandle, MoveAssignmentDeallocatesOnce) {
  MockStreamExecutor executor;
  DeviceMemoryBase memory(&executor, sizeof(executor));
  EXPECT_CALL(executor, Deallocate).Times(1);
  {
    DeviceMemoryHandle releaser(&executor, memory);
    DeviceMemoryHandle releaser2;
    releaser2 = std::move(releaser);
  }
}

}  // namespace stream_executor
