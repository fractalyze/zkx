#include "zkx/math/poly/root_of_unity.h"

#include "gtest/gtest.h"

#include "zkx/math/elliptic_curve/bn/bn254/fq.h"
#include "zkx/math/elliptic_curve/bn/bn254/fr.h"
#include "zkx/math/elliptic_curve/short_weierstrass/test/sw_curve_config.h"
#include "zkx/math/field/babybear/babybear.h"
#include "zkx/math/field/koalabear/koalabear.h"

namespace zkx::math {

using PrimeFieldTypes = testing::Types<
    // clang-format off
    // 8-bit prime fields
    test::Fr,
    // 32-bit prime fields
    Babybear,
    Koalabear,
    // 256-bit prime fields
    bn254::Fq,
    bn254::Fr
    // clang-format on
    >;

namespace {

template <typename PrimeField>
class PrimeFieldBaseTest : public testing::Test {};

}  // namespace

TYPED_TEST_SUITE(PrimeFieldBaseTest, PrimeFieldTypes);

TYPED_TEST(PrimeFieldBaseTest, Decompose) {
  using F = TypeParam;

  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    for (uint32_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      for (uint32_t j = 0; j <= F::Config::kSmallSubgroupAdicity; ++j) {
        uint64_t n = (uint64_t{1} << i) *
                     std::pow(uint64_t{F::Config::kSmallSubgroupBase}, j);

        ASSERT_TRUE(internal::Decompose<F>(n).ok());
      }
    }
  } else {
    GTEST_SKIP() << "No LargeSubgroupRootOfUnity";
  }
}

TYPED_TEST(PrimeFieldBaseTest, TwoAdicRootOfUnity) {
  using F = TypeParam;

  F n = F(2).Pow(F::Config::kTwoAdicity);
  ASSERT_TRUE(F::FromUnchecked(F::Config::kTwoAdicRootOfUnity).Pow(n).IsOne());
}

TYPED_TEST(PrimeFieldBaseTest, LargeSubgroupOfUnity) {
  using F = TypeParam;

  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    F n =
        F(2).Pow(F::Config::kTwoAdicity) *
        F(F::Config::kSmallSubgroupBase).Pow(F::Config::kSmallSubgroupAdicity);
    ASSERT_TRUE(
        F::FromUnchecked(F::Config::kLargeSubgroupRootOfUnity).Pow(n).IsOne());
  } else {
    GTEST_SKIP() << "No LargeSubgroupRootOfUnity";
  }
}

TYPED_TEST(PrimeFieldBaseTest, GetRootOfUnity) {
  using F = TypeParam;

  if constexpr (F::Config::kHasLargeSubgroupRootOfUnity) {
    for (uint32_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      for (uint32_t j = 0; j <= F::Config::kSmallSubgroupAdicity; ++j) {
        uint64_t n = (uint64_t{1} << i) *
                     std::pow(uint64_t{F::Config::kSmallSubgroupBase}, j);
        TF_ASSERT_OK_AND_ASSIGN(F root, GetRootOfUnity<F>(n));
        ASSERT_TRUE(root.Pow(n).IsOne());
      }
    }
  } else if constexpr (F::Config::kHasTwoAdicRootOfUnity) {
    for (uint32_t i = 0; i <= F::Config::kTwoAdicity; ++i) {
      uint64_t n = uint64_t{1} << i;
      TF_ASSERT_OK_AND_ASSIGN(F root, GetRootOfUnity<F>(n));
      ASSERT_TRUE(root.Pow(n).IsOne());
    }
  } else {
    GTEST_SKIP() << "No RootOfUnity";
  }
}

}  // namespace zkx::math
