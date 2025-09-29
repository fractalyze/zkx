#ifndef ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_G1_H_
#define ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_G1_H_

#include "zkx/math/elliptic_curves/bn/bn254/fq.h"
#include "zkx/math/elliptic_curves/bn/bn254/fr.h"
#include "zkx/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "zkx/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "zkx/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "zkx/math/elliptic_curves/short_weierstrass/sw_curve.h"

namespace zkx::math::bn254 {

template <typename BaseField>
class G1SwCurveBaseConfig {
 public:
  constexpr static BaseField kA = 0;
  constexpr static BaseField kB = 3;
  constexpr static BaseField kX = 1;
  constexpr static BaseField kY = 2;
};

class G1SwCurveConfig : public G1SwCurveBaseConfig<Fq> {
 public:
  constexpr static bool kUseMontgomery = true;

  using BaseField = Fq;
  using ScalarField = Fr;
};

using G1Curve = SwCurve<G1SwCurveConfig>;
using G1AffinePoint = zkx::math::AffinePoint<G1Curve>;
using G1JacobianPoint = zkx::math::JacobianPoint<G1Curve>;
using G1PointXyzz = zkx::math::PointXyzz<G1Curve>;

}  // namespace zkx::math::bn254

#endif  // ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_G1_H_
