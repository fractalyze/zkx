// Copyright 2014 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "zkx/base/containers/adapters.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace zkx::base {

TEST(AdaptersTest, Reversed) {
  std::vector<int> v{3, 2, 1};
  int j = 0;
  for (int& i : Reversed(v)) {
    EXPECT_EQ(++j, i);
    i += 100;
  }
  EXPECT_THAT(v, testing::ContainerEq(std::vector<int>{103, 102, 101}));
}

TEST(AdaptersTest, ReversedArray) {
  int v[3] = {3, 2, 1};
  int j = 0;
  for (int& i : Reversed(v)) {
    EXPECT_EQ(++j, i);
    i += 100;
  }
  EXPECT_THAT(v, testing::ElementsAreArray({103, 102, 101}));
}

TEST(AdaptersTest, ReversedConst) {
  std::vector<int> v{3, 2, 1};
  const std::vector<int>& cv = v;
  int j = 0;
  for (int i : Reversed(cv)) {
    EXPECT_EQ(++j, i);
  }
}

}  // namespace zkx::base
