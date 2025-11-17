#ifndef ZKX_MATH_ELLIPTIC_CURVE_BN_BN254_CURVE_H_
#define ZKX_MATH_ELLIPTIC_CURVE_BN_BN254_CURVE_H_

#include "zkx/math/elliptic_curve/bn/bn254/g1.h"
#include "zkx/math/elliptic_curve/bn/bn254/g2.h"

namespace zkx::math::bn254 {

class CurveStd {
 public:
  using G1Curve = bn254::G1CurveStd;
  using G2Curve = bn254::G2CurveStd;

  using StdConfig = CurveStd;
};

class Curve {
 public:
  using G1Curve = bn254::G1Curve;
  using G2Curve = bn254::G2Curve;

  using StdCurve = CurveStd;
};

}  // namespace zkx::math::bn254

#endif  // ZKX_MATH_ELLIPTIC_CURVE_BN_BN254_CURVE_H_
