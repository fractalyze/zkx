#ifndef ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FQ2_H_
#define ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FQ2_H_

#include <stdint.h>

#include "zkx/math/base/extension_field.h"
#include "zkx/math/elliptic_curves/bn/bn254/fq.h"

namespace zkx::math::bn254 {

template <typename BaseField>
class Fq2BaseConfig {
 public:
  constexpr static uint32_t kDegreeOverBaseField = 2;
  constexpr static BaseField kNonResidue = -1;
};

class Fq2Config : public Fq2BaseConfig<Fq> {
 public:
  constexpr static bool kUseMontgomery = true;

  using BaseField = Fq;
  using BasePrimeField = Fq;
};

using Fq2 = ExtensionField<Fq2Config>;

}  // namespace zkx::math::bn254

#endif  // ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FQ2_H_
