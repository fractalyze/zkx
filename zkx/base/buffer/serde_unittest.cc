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

#include "zkx/base/buffer/serde.h"

#include "gtest/gtest.h"

namespace zkx::base {

enum class Color {
  kRed,
  kBlue,
  kGreen,
};

TEST(SerdeTest, BuiltInSerdeTest) {
#define TEST_BUILTIN_TYPES(type)                            \
  EXPECT_TRUE(base::internal::IsBuiltinSerde<type>::value); \
  EXPECT_FALSE(base::internal::IsNonBuiltinSerde<type>::value)

#define TEST_NON_BUILTIN_TYPES(type)                         \
  EXPECT_FALSE(base::internal::IsBuiltinSerde<type>::value); \
  EXPECT_TRUE(base::internal::IsNonBuiltinSerde<type>::value)

  TEST_BUILTIN_TYPES(bool);
  TEST_BUILTIN_TYPES(char);
  TEST_BUILTIN_TYPES(uint16_t);
  TEST_BUILTIN_TYPES(uint32_t);
  TEST_BUILTIN_TYPES(uint64_t);
  TEST_BUILTIN_TYPES(int16_t);
  TEST_BUILTIN_TYPES(int32_t);
  TEST_BUILTIN_TYPES(int64_t);

  TEST_NON_BUILTIN_TYPES(Color);
  TEST_NON_BUILTIN_TYPES(std::string_view);
  TEST_NON_BUILTIN_TYPES(std::string);
  TEST_NON_BUILTIN_TYPES(uint64_t[4]);
  TEST_NON_BUILTIN_TYPES(std::vector<uint64_t>);

  using Array = std::array<uint64_t, 4>;
  TEST_NON_BUILTIN_TYPES(Array);

  using Tuple = std::tuple<uint64_t, uint64_t>;
  TEST_NON_BUILTIN_TYPES(Tuple);

#undef TEST_NON_BUILTIN_TYPES
#undef TEST_BUILTIN_TYPES
}

}  // namespace zkx::base
