#include "zkx/math/base/prime_field.h"

#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/math/elliptic_curves/bn/bn254/fr.h"

namespace zkx::math::bn254 {

TEST(PrimeFieldTest, Zero) {
  EXPECT_TRUE(Fr::Zero().IsZero());
  EXPECT_FALSE(Fr::One().IsZero());
}

TEST(PrimeFieldTest, One) {
  EXPECT_TRUE(Fr::One().IsOne());
  EXPECT_FALSE(Fr::Zero().IsOne());
}

TEST(PrimeFieldTest, Operations) {
  Fr a = *Fr::FromHexString(
      "0xb94db59332f8a619901d39188315c421beafb516eb8a3ab56ceed7df960ede2");
  Fr b = *Fr::FromHexString(
      "0xecec51689891ef3f7ff39040036fd0e282687d392abe8f011589f1c755500a2");
  // clang-format off
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(a != b);
  EXPECT_EQ(a + b, *Fr::FromHexString("0x1a63a06fbcb8a95591010c95886859504411832501648c9b68278c9a6eb5ee84"));
  EXPECT_EQ(a.Double(), *Fr::FromHexString("0x1729b6b2665f14c33203a7231062b88437d5f6a2dd714756ad9ddafbf2c1dbc4"));
  EXPECT_EQ(a - b, *Fr::FromHexString("0x2d2a64b58ad80b975952e044097bb7911bf85bc655c62b4c895843f5740bed41"));
  EXPECT_EQ(a * b, *Fr::FromHexString("0x30593207bceeeba352060f9f1a3ae9d2214f428a90ad235867aabd7a10640d44"));
  EXPECT_EQ(a.Square(), *Fr::FromHexString("0x2e3797fa80f1e71d9b23f1a6a2572f6aa2de416a1b31ceca88ef28944fd292a"));
  TF_ASSERT_OK_AND_ASSIGN(Fr a_inverse, a.Inverse());
  EXPECT_TRUE((a * a_inverse).IsOne());
  // clang-format on
}

}  // namespace zkx::math::bn254
