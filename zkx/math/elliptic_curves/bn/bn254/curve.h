#ifndef ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_H_
#define ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_H_

#include "zkx/math/elliptic_curves/bn/bn254/g1.h"
#include "zkx/math/elliptic_curves/bn/bn254/g2.h"

namespace zkx::math::bn254 {

class Curve {
 public:
  using G1Curve = bn254::G1Curve;
  using G2Curve = bn254::G2Curve;
};

}  // namespace zkx::math::bn254

#endif  // ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_CURVE_H_
