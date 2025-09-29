#ifndef ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_G2_H_
#define ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_G2_H_

#include "zkx/math/elliptic_curves/bn/bn254/fq2.h"
#include "zkx/math/elliptic_curves/bn/bn254/fr.h"
#include "zkx/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "zkx/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "zkx/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "zkx/math/elliptic_curves/short_weierstrass/sw_curve.h"

namespace zkx::math::bn254 {

template <typename BaseField>
class G2SwCurveBaseConfig {
 public:
  constexpr static BaseField kA = {{0, 0}};
  constexpr static BaseField kB = {{
                                       UINT64_C(3632125457679333605),
                                       UINT64_C(13093307605518643107),
                                       UINT64_C(9348936922344483523),
                                       UINT64_C(3104278944836790958),
                                   },
                                   {
                                       UINT64_C(16474938222586303954),
                                       UINT64_C(12056031220135172178),
                                       UINT64_C(14784384838321896948),
                                       UINT64_C(42524369107353300),
                                   }};
  constexpr static BaseField kX = {{
                                       UINT64_C(5106727233969649389),
                                       UINT64_C(7440829307424791261),
                                       UINT64_C(4785637993704342649),
                                       UINT64_C(1729627375292849782),
                                   },
                                   {
                                       UINT64_C(10945020018377822914),
                                       UINT64_C(17413811393473931026),
                                       UINT64_C(8241798111626485029),
                                       UINT64_C(1841571559660931130),
                                   }};
  constexpr static BaseField kY = {{
                                       UINT64_C(5541340697920699818),
                                       UINT64_C(16416156555105522555),
                                       UINT64_C(5380518976772849807),
                                       UINT64_C(1353435754470862315),
                                   },
                                   {
                                       UINT64_C(6173549831154472795),
                                       UINT64_C(13567992399387660019),
                                       UINT64_C(17050234209342075797),
                                       UINT64_C(650358724130500725),
                                   }};
};

class G2SwCurveConfig : public G2SwCurveBaseConfig<Fq2> {
 public:
  constexpr static bool kUseMontgomery = true;

  using BaseField = Fq2;
  using ScalarField = Fr;
};

using G2Curve = SwCurve<G2SwCurveConfig>;
using G2AffinePoint = zkx::math::AffinePoint<G2Curve>;
using G2JacobianPoint = zkx::math::JacobianPoint<G2Curve>;
using G2PointXyzz = zkx::math::PointXyzz<G2Curve>;

}  // namespace zkx::math::bn254

#endif  // ZKX_MATH_ELLIPTIC_CURVES_BN_BN254_G2_H_
