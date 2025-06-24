#include "zkx/math/base/big_int.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"

namespace zkx::math {

using ::tsl::testing::StatusIs;

TEST(BigIntTest, Zero) {
  BigInt<2> big_int = BigInt<2>::Zero();
  EXPECT_TRUE(big_int.IsZero());
  EXPECT_FALSE(big_int.IsOne());
}

TEST(BigIntTest, One) {
  BigInt<2> big_int = BigInt<2>::One();
  EXPECT_FALSE(big_int.IsZero());
  EXPECT_TRUE(big_int.IsOne());
}

TEST(BigIntTest, DecString) {
  // 1 << 65
  absl::StatusOr<BigInt<2>> big_int =
      BigInt<2>::FromDecString("36893488147419103232");
  TF_ASSERT_OK(big_int);
  EXPECT_EQ(big_int->ToString(), "36893488147419103232");

  // Invalid input
  EXPECT_THAT(BigInt<2>::FromDecString("x"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // 1 << 128
  EXPECT_THAT(
      BigInt<2>::FromDecString("340282366920938463463374607431768211456"),
      StatusIs(absl::StatusCode::kOutOfRange));
}

TEST(BigIntTest, HexString) {
  {
    // 1 << 65
    absl::StatusOr<BigInt<2>> big_int =
        BigInt<2>::FromHexString("20000000000000000");
    TF_ASSERT_OK(big_int);
    EXPECT_EQ(big_int->ToHexString(), "0x20000000000000000");
  }
  {
    // 1 << 65
    absl::StatusOr<BigInt<2>> big_int =
        BigInt<2>::FromHexString("0x20000000000000000");
    TF_ASSERT_OK(big_int);
    EXPECT_EQ(big_int->ToHexString(), "0x20000000000000000");
  }

  // Invalid input
  EXPECT_THAT(BigInt<2>::FromDecString("g"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // 1 << 128
  EXPECT_THAT(BigInt<2>::FromHexString("0x100000000000000000000000000000000"),
              StatusIs(absl::StatusCode::kOutOfRange));
}

TEST(BigIntTest, Comparison) {
  // 1 << 65
  BigInt<2> big_int = *BigInt<2>::FromHexString("20000000000000000");
  BigInt<2> big_int2 = *BigInt<2>::FromHexString("20000000000000001");
  EXPECT_TRUE(big_int == big_int);
  EXPECT_TRUE(big_int != big_int2);
  EXPECT_TRUE(big_int < big_int2);
  EXPECT_TRUE(big_int <= big_int2);
  EXPECT_TRUE(big_int2 > big_int);
  EXPECT_TRUE(big_int2 >= big_int);
}

TEST(BigIntTest, Operations) {
  BigInt<2> a =
      *BigInt<2>::FromDecString("123456789012345678909876543211235312");
  BigInt<2> b =
      *BigInt<2>::FromDecString("734581237591230158128731489729873983");

  EXPECT_EQ(a + b,
            *BigInt<2>::FromDecString("858038026603575837038608032941109295"));
  EXPECT_EQ(a << 1,
            *BigInt<2>::FromDecString("246913578024691357819753086422470624"));
  EXPECT_EQ(a >> 1,
            *BigInt<2>::FromDecString("61728394506172839454938271605617656"));
  EXPECT_EQ(a - b, *BigInt<2>::FromDecString(
                       "339671242472359578984155752485249572785"));
  EXPECT_EQ(b - a,
            *BigInt<2>::FromDecString("611124448578884479218854946518638671"));
  EXPECT_EQ(a * b, *BigInt<2>::FromDecString(
                       "335394729415762779748307316131549975568"));
  BigInt<2> divisor(123456789);
  EXPECT_EQ(*(a / divisor),
            *BigInt<2>::FromDecString("1000000000100000000080000000"));
  EXPECT_EQ(*(a % divisor), BigInt<2>(91235312));
}

}  // namespace zkx::math
