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

#include "zkx/math/field/prime_field_serde.h"

#include "gtest/gtest.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fq.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/fr.h"
#include "zk_dtypes/include/elliptic_curve/short_weierstrass/test/sw_curve_config.h"
#include "zk_dtypes/include/field/babybear/babybear.h"
#include "zk_dtypes/include/field/goldilocks/goldilocks.h"
#include "zk_dtypes/include/field/koalabear/koalabear.h"
#include "zk_dtypes/include/field/mersenne31/mersenne31.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/base/auto_reset.h"
#include "zkx/base/buffer/vector_buffer.h"

namespace zkx {
namespace {

template <typename T>
class PrimeFieldTypedTest : public testing::Test {};

using PrimeFieldTypes = testing::Types<
    // clang-format off
    // 8-bit prime fields
    zk_dtypes::test::Fr,
    zk_dtypes::test::FrStd,
    // 32-bit prime fields
    zk_dtypes::Babybear,
    zk_dtypes::BabybearStd,
    zk_dtypes::Koalabear,
    zk_dtypes::Mersenne31,
    // 64-bit prime fields
    zk_dtypes::Goldilocks,
    // 256-bit prime fields
    zk_dtypes::bn254::Fq,
    zk_dtypes::bn254::FqStd,
    zk_dtypes::bn254::Fr
    // clang-format on
    >;
TYPED_TEST_SUITE(PrimeFieldTypedTest, PrimeFieldTypes);

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

}  // namespace
}  // namespace zkx
