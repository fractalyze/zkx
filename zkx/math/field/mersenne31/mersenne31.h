#ifndef ZKX_MATH_FIELD_MERSENNE31_MERSENNE31_H_
#define ZKX_MATH_FIELD_MERSENNE31_MERSENNE31_H_

#include <cstdint>

#include "zkx/math/field/small_prime_field.h"

namespace zkx::math {

struct Mersenne31BaseConfig {
  constexpr static size_t kModulusBits = 31;
  constexpr static uint32_t kModulus = 2147483647;

  constexpr static uint32_t kRSquared = 4;
  constexpr static uint32_t kNPrime = 2147483647;

  constexpr static uint32_t kTwoAdicity = 1;

  constexpr static uint32_t kTrace = 1073741823;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct Mersenne31StdConfig : public Mersenne31BaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = Mersenne31StdConfig;

  constexpr static uint32_t kOne = 1;

  constexpr static uint32_t kTwoAdicRootOfUnity = 2147483646;
};

struct Mersenne31Config : public Mersenne31BaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Mersenne31StdConfig;

  constexpr static uint32_t kOne = 2;

  constexpr static uint32_t kTwoAdicRootOfUnity = 2147483645;
};

using Mersenne31 = PrimeField<Mersenne31Config>;
using Mersenne31Std = PrimeField<Mersenne31StdConfig>;

}  // namespace zkx::math

#endif  // ZKX_MATH_FIELD_MERSENNE31_MERSENNE31_H_
