/* Copyright 2017 The OpenXLA Authors.

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

#ifndef ZKX_TESTS_LITERAL_TEST_UTIL_H_
#define ZKX_TESTS_LITERAL_TEST_UTIL_H_

#include "gtest/gtest.h"

#include "zkx/literal.h"

namespace zkx {

// Utility class for making expectations/assertions related to ZKX literals.
class LiteralTestUtil {
 public:
  [[nodiscard]] static ::testing::AssertionResult Equal(
      const LiteralSlice& expected, const LiteralSlice& actual);
};

}  // namespace zkx

#endif  // ZKX_TESTS_LITERAL_TEST_UTIL_H_
