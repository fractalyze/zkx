/* Copyright 2019 The OpenXLA Authors.
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

#include "zkx/stream_executor/host/host_stream.h"

#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/stream_executor/platform_manager.h"

namespace stream_executor {

TEST(HostStream, EnforcesFIFOOrder) {
  Platform* platform = PlatformManager::PlatformWithName("Host").value();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  absl::Mutex mu;
  int expected = 0;
  bool ok = true;
  for (int i = 0; i < 2000; ++i) {
    ASSERT_TRUE(stream
                    ->DoHostCallback([i, &mu, &expected, &ok]() {
                      absl::MutexLock lock(&mu);
                      if (expected != i) {
                        ok = false;
                      }
                      ++expected;
                    })
                    .ok());
  }
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  absl::MutexLock lock(&mu);
  EXPECT_TRUE(ok);
}

TEST(HostStream, ReportsHostCallbackError) {
  Platform* platform = PlatformManager::PlatformWithName("Host").value();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  ASSERT_TRUE(stream
                  ->DoHostCallbackWithStatus(
                      []() { return absl::InternalError("error!"); })
                  .ok());

  auto status = stream->BlockHostUntilDone();
  ASSERT_EQ(status.code(), absl::StatusCode::kInternal);
  ASSERT_EQ(status.message(), "error!");
}

TEST(HostStream, ReportsFirstHostCallbackError) {
  Platform* platform = PlatformManager::PlatformWithName("Host").value();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  ASSERT_TRUE(stream
                  ->DoHostCallbackWithStatus(
                      []() { return absl::InternalError("error 1"); })
                  .ok());
  ASSERT_TRUE(stream
                  ->DoHostCallbackWithStatus(
                      []() { return absl::InternalError("error 2"); })
                  .ok());

  // "error 2" is just lost.
  ASSERT_EQ(stream->BlockHostUntilDone().message(), "error 1");
}

}  // namespace stream_executor
