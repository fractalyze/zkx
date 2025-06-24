#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/math/elliptic_curves/bn/bn254/fq2.h"

namespace zkx::math::bn254 {

TEST(ExtensionFieldTest, Zero) {
  EXPECT_TRUE(Fq2::Zero().IsZero());
  EXPECT_FALSE(Fq2::One().IsZero());
}

TEST(ExtensionFieldTest, One) {
  EXPECT_TRUE(Fq2::One().IsOne());
  EXPECT_FALSE(Fq2::Zero().IsOne());
}

TEST(ExtensionFieldTest, Operations) {
  Fq2 a = {
      *Fq::FromHexString(
          "0xb94db59332f8a619901d39188315c421beafb516eb8a3ab56ceed7df960ede2"),
      *Fq::FromHexString(
          "0xecec51689891ef3f7ff39040036fd0e282687d392abe8f011589f1c755500a2"),
  };
  Fq2 b = {
      *Fq::FromHexString(
          "0x2fd5bb90b6043853a651f3e930e7c549f448e5562cbc823bbe2a84a67830e1d"),
      *Fq::FromHexString(
          "0x284f3562b7cda2c9f329ac7019d4778a9bfb820f4639687e4573eadfed505d97"),
  };
  // clang-format off
  EXPECT_EQ(a + b, Fq2({
      *Fq::FromHexString(
          "0xe9237123e8fcde6d366f2d01b3fd896bb2f89a6d1846bcf12b195c860e3fbff"),
      *Fq::FromHexString(
          "0x6b9ac066025219432d89fbd988a1c3b2ca09f51707386e11aabfde58a2860f2"),
  }));
  EXPECT_EQ(a.Double(), Fq2({
      *Fq::FromHexString(
          "0x1729b6b2665f14c33203a7231062b88437d5f6a2dd714756ad9ddafbf2c1dbc4"),
      *Fq::FromHexString(
          "0x1d9d8a2d13123de7effe7208006dfa1c504d0fa72557d1e022b13e38eaaa0144"),
  }));
  EXPECT_EQ(a - b, Fq2({
      *Fq::FromHexString(
          "0x8977fa027cf46dc5e9cb452f522dfed7ca66cfc0becdb879aec453391dddfc5"),
      *Fq::FromHexString(
          "0x16e3de26b2ed1c53bd25d24a67e3dde123ac7055b4e44aff080540536081a052"),
  }));
  EXPECT_EQ(a * b, Fq2({
      *Fq::FromHexString(
          "0x25908f85559394401b8ced51130b426c319a22feb86b4b36e2b5aa6de4ab5ebf"),
      *Fq::FromHexString(
          "0x283abe3026369b6da48074048233145a1c88dce48308e952c948eff8e8f0bc51"),
  }));
  EXPECT_EQ(a.Square(), a * a);
  TF_ASSERT_OK_AND_ASSIGN(Fq2 a_inverse, a.Inverse());
  EXPECT_TRUE((a * a_inverse).IsOne());
  // clang-format on
}

}  // namespace zkx::math::bn254
