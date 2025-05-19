#ifndef ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_
#define ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_

#include "zkx/math/base/prime_field.h"

namespace zkx::math::bn254 {

struct FrConfig {
  constexpr static size_t kModulusBits = 254;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(4891460686036598785),
      UINT64_C(2896914383306846353),
      UINT64_C(13281191951274694749),
      UINT64_C(3486998266802970665),
  };
  constexpr static BigInt<4> kOne = {
      UINT64_C(12436184717236109307),
      UINT64_C(3962172157175319849),
      UINT64_C(7381016538464732718),
      UINT64_C(1011752739694698287),
  };

  constexpr static BigInt<4> kRSquared = {
      UINT64_C(1997599621687373223),
      UINT64_C(6052339484930628067),
      UINT64_C(10108755138030829701),
      UINT64_C(150537098327114917),
  };
  constexpr static uint64_t kNPrime = UINT64_C(14042775128853446655);
};

using Fr = PrimeField<FrConfig>;

}  // namespace zkx::math::bn254

#endif  // ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_
