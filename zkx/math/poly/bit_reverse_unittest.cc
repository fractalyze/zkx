/* Copyright 2025 The ZKX Authors.

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

#include "zkx/math/poly/bit_reverse.h"

#include "gtest/gtest.h"

namespace zkx::math {

TEST(BitReverseTest, BitReverse64_Symmetry) {
  // Test that bit reverse is its own inverse
  uint64_t test_values[] = {0x123456789ABCDEF0, 0xAAAAAAAAAAAAAAAA,
                            0x5555555555555555, 0xFFFFFFFFFFFFFFFF};

  for (uint64_t val : test_values) {
    EXPECT_EQ(BitReverse64(BitReverse64(val)), val);
  }
}

TEST(BitReverseTest, BitReverse) {
  // Test BitReverse function with different bit lengths
  EXPECT_EQ(BitReverse(0, 1), 0);
  EXPECT_EQ(BitReverse(1, 1), 1);

  EXPECT_EQ(BitReverse(0, 2), 0);
  EXPECT_EQ(BitReverse(1, 2), 2);
  EXPECT_EQ(BitReverse(2, 2), 1);
  EXPECT_EQ(BitReverse(3, 2), 3);

  EXPECT_EQ(BitReverse(0, 3), 0);
  EXPECT_EQ(BitReverse(1, 3), 4);
  EXPECT_EQ(BitReverse(2, 3), 2);
  EXPECT_EQ(BitReverse(3, 3), 6);
  EXPECT_EQ(BitReverse(4, 3), 1);
  EXPECT_EQ(BitReverse(5, 3), 5);
  EXPECT_EQ(BitReverse(6, 3), 3);
  EXPECT_EQ(BitReverse(7, 3), 7);
}

TEST(BitReverseTest, BitReverseShuffleInPlace) {
  // Test BitReverseShuffleInPlace with small vectors
  std::vector<int> v1 = {0, 1, 2, 3};
  std::vector<int> expected1 = {0, 2, 1, 3};  // log2(4) = 2
  BitReverseShuffleInPlace(v1);
  EXPECT_EQ(v1, expected1);

  std::vector<int> v2 = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> expected2 = {0, 4, 2, 6, 1, 5, 3, 7};  // log2(8) = 3
  BitReverseShuffleInPlace(v2);
  EXPECT_EQ(v2, expected2);
}

TEST(BitReverseTest, BitReverseShuffle) {
  // Test BitReverseShuffle function
  std::vector<int> input = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> expected = {0, 4, 2, 6, 1, 5, 3, 7};
  std::vector<int> result = BitReverseShuffle(input);
  EXPECT_EQ(result, expected);

  // Test with power of 2 size
  std::vector<int> input2 = {0, 1, 2, 3};
  std::vector<int> expected2 = {0, 2, 1, 3};
  std::vector<int> result2 = BitReverseShuffle(input2);
  EXPECT_EQ(result2, expected2);
}

TEST(BitReverseTest, BitReverseShuffle_EdgeCases) {
  // Test edge cases
  std::vector<int> empty = {};
  EXPECT_EQ(BitReverseShuffle(empty), empty);

  std::vector<int> single = {42};
  EXPECT_EQ(BitReverseShuffle(single), single);

  std::vector<int> two = {1, 2};
  EXPECT_EQ(BitReverseShuffle(two), two);
}

}  // namespace zkx::math
