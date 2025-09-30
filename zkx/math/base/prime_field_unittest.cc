#include "zkx/math/base/prime_field.h"

#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/base/auto_reset.h"
#include "zkx/base/buffer/vector_buffer.h"
#include "zkx/math/elliptic_curves/bn/bn254/fq.h"
#include "zkx/math/elliptic_curves/bn/bn254/fr.h"

namespace zkx::math {
namespace bn254 {

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
  EXPECT_EQ(a.Pow(30), *Fr::FromHexString("0xa5c969115bc5da7d6bfe244ec24b7e244d454561569de7acf0980633533fcca"));
  // clang-format on
}

}  // namespace bn254

template <typename T>
class PrimeFieldTypedTest : public testing::Test {};

using PrimeFieldTypes = testing::Types<bn254::Fq, bn254::Fr>;
TYPED_TEST_SUITE(PrimeFieldTypedTest, PrimeFieldTypes);

TYPED_TEST(PrimeFieldTypedTest, Zero) {
  using F = TypeParam;
  EXPECT_TRUE(F::Zero().IsZero());
  EXPECT_FALSE(F::One().IsZero());
}

TYPED_TEST(PrimeFieldTypedTest, One) {
  using F = TypeParam;

  EXPECT_TRUE(F::One().IsOne());
  EXPECT_FALSE(F::Zero().IsOne());
}

TYPED_TEST(PrimeFieldTypedTest, SquareRoot) {
  using F = TypeParam;

  F a = F::Random();
  F a2 = a.Square();
  TF_ASSERT_OK_AND_ASSIGN(F sqrt, a2.SquareRoot());
  EXPECT_TRUE(a == sqrt || a == -sqrt);
}

TYPED_TEST(PrimeFieldTypedTest, Inverse) {
  using F = TypeParam;

  F a = F::Random();
  while (a.IsZero()) {
    a = F::Random();
  }
  TF_ASSERT_OK_AND_ASSIGN(F a_inv, a.Inverse());
  EXPECT_TRUE((a * a_inv).IsOne());
}

TYPED_TEST(PrimeFieldTypedTest, Serde) {
  using F = TypeParam;

  F expected = F::Random();

  for (size_t i = 0; i < 2; ++i) {
    bool s_is_in_montgomery = i == 0;
    SCOPED_TRACE(
        absl::Substitute("s_is_in_montgomery: $0", s_is_in_montgomery));
    base::AutoReset<bool> auto_reset(&base::Serde<F>::s_is_in_montgomery,
                                     s_is_in_montgomery);
    base::Uint8VectorBuffer write_buf;
    TF_ASSERT_OK(write_buf.Grow(base::EstimateSize(expected)));
    TF_ASSERT_OK(write_buf.Write(expected));
    ASSERT_TRUE(write_buf.Done());

    write_buf.set_buffer_offset(0);

    F value;
    TF_ASSERT_OK(write_buf.Read(&value));
    EXPECT_EQ(expected, value);
  }
}

TYPED_TEST(PrimeFieldTypedTest, JsonSerde) {
  using F = TypeParam;

  rapidjson::Document doc;

  F expected = F::Random();
  for (size_t i = 0; i < 2; ++i) {
    bool s_is_in_montgomery = i == 0;
    SCOPED_TRACE(
        absl::Substitute("s_is_in_montgomery: $0", s_is_in_montgomery));
    base::AutoReset<bool> auto_reset(&base::JsonSerde<F>::s_is_in_montgomery,
                                     s_is_in_montgomery);
    rapidjson::Value json_value =
        base::JsonSerde<F>::From(expected, doc.GetAllocator());
    TF_ASSERT_OK_AND_ASSIGN(F value, base::JsonSerde<F>::To(json_value, ""));
    EXPECT_EQ(expected, value);
  }
}

}  // namespace zkx::math
