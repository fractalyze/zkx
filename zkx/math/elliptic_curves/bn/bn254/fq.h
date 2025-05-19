#ifndef ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_
#define ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_

#include "zkx/math/base/prime_field.h"

namespace zkx::math::bn254 {

struct FqConfig {
  constexpr static size_t kModulusBits = 254;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(4332616871279656263),
      UINT64_C(10917124144477883021),
      UINT64_C(13281191951274694749),
      UINT64_C(3486998266802970665),
  };
  constexpr static BigInt<4> kOne = {
      UINT64_C(15230403791020821917),
      UINT64_C(754611498739239741),
      UINT64_C(7381016538464732716),
      UINT64_C(1011752739694698287),
  };

  constexpr static BigInt<4> kRSquared = {
      UINT64_C(17522657719365597833),
      UINT64_C(13107472804851548667),
      UINT64_C(5164255478447964150),
      UINT64_C(493319470278259999),
  };
  constexpr static uint64_t kNPrime = UINT64_C(9786893198990664585);
};

using Fq = PrimeField<FqConfig>;

}  // namespace zkx::math::bn254

#endif  // ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FQ_H_
