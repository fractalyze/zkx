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
