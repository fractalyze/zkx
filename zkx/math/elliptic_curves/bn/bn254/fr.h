#ifndef ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_
#define ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_

#include "zkx/math/base/prime_field.h"

namespace zkx::math::bn254 {

struct FrBaseConfig {
  constexpr static size_t kModulusBits = 254;
  constexpr static BigInt<4> kModulus = {
      UINT64_C(4891460686036598785),
      UINT64_C(2896914383306846353),
      UINT64_C(13281191951274694749),
      UINT64_C(3486998266802970665),
  };

  constexpr static BigInt<4> kRSquared = {
      UINT64_C(1997599621687373223),
      UINT64_C(6052339484930628067),
      UINT64_C(10108755138030829701),
      UINT64_C(150537098327114917),
  };
  constexpr static uint64_t kNPrime = UINT64_C(14042775128853446655);

  constexpr static uint32_t kTwoAdicity = 28;
  constexpr static uint32_t kSmallSubgroupBase = 3;
  constexpr static uint32_t kSmallSubgroupAdicity = 2;

  constexpr static BigInt<4> kTrace = {
      UINT64_C(11211439779908376895),
      UINT64_C(1735440370612733063),
      UINT64_C(1376415503089949544),
      UINT64_C(12990080814),
  };

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = true;
};
struct FrConfig : public FrBaseConfig {
  constexpr static bool kUseMontgomery = true;

  constexpr static BigInt<4> kOne = {
      UINT64_C(12436184717236109307),
      UINT64_C(3962172157175319849),
      UINT64_C(7381016538464732718),
      UINT64_C(1011752739694698287),
  };

  constexpr static BigInt<4> kTwoAdicRootOfUnity = {
      UINT64_C(7164790868263648668),
      UINT64_C(11685701338293206998),
      UINT64_C(6216421865291908056),
      UINT64_C(1756667274303109607),
  };

  constexpr static BigInt<4> kLargeSubgroupRootOfUnity = {
      UINT64_C(13572693633482797243),
      UINT64_C(9300120515097308470),
      UINT64_C(4644696547510559629),
      UINT64_C(1568669608320580247),
  };
};

using Fr = PrimeField<FrConfig>;

}  // namespace zkx::math::bn254

#endif  // ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FR_H_
