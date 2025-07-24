#ifndef ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FQ2_H_
#define ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FQ2_H_

#include <stdint.h>

#include "zkx/math/base/extension_field.h"
#include "zkx/math/elliptic_curves/bn/bn254/fq.h"

namespace zkx::math::bn254 {

class Fq2Config {
 public:
  using BaseField = Fq;
  using BasePrimeField = Fq;

  constexpr static uint32_t kDegreeOverBaseField = 2;
  constexpr static BaseField kNonResidue = -1;
};

using Fq2 = ExtensionField<Fq2Config>;

}  // namespace zkx::math::bn254

#endif  // ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_FQ2_H_
