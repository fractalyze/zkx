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

#include "zkx/python/ifrt/future.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"

namespace zkx::ifrt {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(FutureTest, JoinZeroFuture) {
  Future<> future = JoinFutures({});

  TF_EXPECT_OK(future.Await());
}

TEST(FutureTest, JoinOneOkFuture) {
  Promise<> promise = Future<>::CreatePromise();
  std::vector<Future<>> futures;
  futures.push_back(Future<>(promise));

  Future<> future = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(future.IsReady());
  promise.Set(absl::OkStatus());
  TF_EXPECT_OK(future.Await());
}

TEST(FutureTest, JoinOneFailingFuture) {
  Promise<> promise = Future<>::CreatePromise();
  std::vector<Future<>> futures;
  futures.push_back(Future<>(promise));

  Future<> future = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(future.IsReady());
  promise.Set(absl::InvalidArgumentError("Some error"));
  EXPECT_THAT(future.Await(), StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Some error")));
}

TEST(FutureTest, JoinAllOkFutures) {
  constexpr int kNumFutures = 3;
  std::vector<Promise<>> promises;
  std::vector<Future<>> futures;
  promises.reserve(kNumFutures);
  futures.reserve(kNumFutures);
  for (int i = 0; i < kNumFutures; ++i) {
    promises.push_back(Future<>::CreatePromise());
    futures.push_back(Future<>(promises.back()));
  }

  Future<> future = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(future.IsReady());
  for (Promise<>& promise : promises) {
    promise.Set(absl::OkStatus());
  }
  TF_EXPECT_OK(future.Await());
}

TEST(FutureTest, JoinAllFailingFutures) {
  constexpr int kNumFutures = 3;
  std::vector<Promise<>> promises;
  std::vector<Future<>> futures;
  promises.reserve(kNumFutures);
  futures.reserve(kNumFutures);
  for (int i = 0; i < kNumFutures; ++i) {
    promises.push_back(Future<>::CreatePromise());
    futures.push_back(Future<>(promises.back()));
  }

  Future<> future = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(future.IsReady());
  for (Promise<>& promise : promises) {
    promise.Set(absl::InvalidArgumentError("Some error"));
  }
  EXPECT_THAT(future.Await(), StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Some error")));
}

class JoinAllOkFuturesExceptForOneTest : public testing::TestWithParam<int> {};

TEST_P(JoinAllOkFuturesExceptForOneTest, JoinAllOkFuturesExceptForOne) {
  const int kNumFutures = 3;
  const int failing_future_idx = GetParam();
  std::vector<Promise<>> promises;
  std::vector<Future<>> futures;
  promises.reserve(kNumFutures);
  futures.reserve(kNumFutures);
  for (int i = 0; i < kNumFutures; ++i) {
    promises.push_back(Future<>::CreatePromise());
    futures.push_back(Future<>(promises.back()));
  }

  Future<> future = JoinFutures(absl::MakeSpan(futures));

  ASSERT_FALSE(future.IsReady());
  for (int i = 0; i < kNumFutures; ++i) {
    if (i == failing_future_idx) {
      promises[i].Set(absl::InvalidArgumentError("Some error"));
    } else {
      promises[i].Set(absl::OkStatus());
    }
  }
  EXPECT_THAT(future.Await(), StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Some error")));
}

INSTANTIATE_TEST_SUITE_P(FutureTest, JoinAllOkFuturesExceptForOneTest,
                         testing::Range(0, 3));

}  // namespace
}  // namespace zkx::ifrt
