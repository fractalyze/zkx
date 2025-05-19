#ifndef ZKX_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_H_
#define ZKX_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_H_

#include "zkx/math/geometry/curve_type.h"
#include "zkx/math/geometry/point_declarations.h"

namespace zkx::math {

// Curve for Short Weierstrass model.
// See https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html for more details.
// This config represents y² = x³ + a * x + b, where a and b are constants.
template <typename _Config>
class SwCurve {
 public:
  using Config = _Config;

  using BaseField = typename Config::BaseField;
  using ScalarField = typename Config::ScalarField;
  using AffinePoint = zkx::math::AffinePoint<SwCurve<Config>>;
  using ProjectivePoint = zkx::math::AffinePoint<SwCurve<Config>>;
  using PointXyzz = zkx::math::AffinePoint<SwCurve<Config>>;

  constexpr static CurveType kType = CurveType::kShortWeierstrass;
};

}  // namespace zkx::math

#endif  // ZKX_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_H_
