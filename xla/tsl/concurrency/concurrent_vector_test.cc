/* Copyright 2022 Google LLC. All Rights Reserved.

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

#include "xla/tsl/concurrency/concurrent_vector.h"

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

namespace tsl {

using ::tsl::internal::ConcurrentVector;

TEST(ConcurrentVectorTest, SingleThreaded) {
  ConcurrentVector<int> vec(1);

  constexpr int kCount = 1000;

  for (int i = 0; i < kCount; ++i) {
    ASSERT_EQ(i, vec.emplace_back(i));
  }

  for (int i = 0; i < kCount; ++i) {
    EXPECT_EQ(i, vec[i]);
  }
}

// TODO(chokobole): Add OneWriterOneReader, TwoWritersTwoReaders test cases.
// Dependency: ThreadPool

}  // namespace tsl
