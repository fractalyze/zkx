#ifndef ZKX_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_H_
#define ZKX_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_H_

#include "absl/status/statusor.h"

#include "xla/tsl/platform/statusor.h"
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
  using JacobianPoint = zkx::math::JacobianPoint<SwCurve<Config>>;
  using PointXyzz = zkx::math::PointXyzz<SwCurve<Config>>;

  constexpr static CurveType kType = CurveType::kShortWeierstrass;

  // Attempts to construct an affine point given an x-coordinate.
  constexpr static absl::StatusOr<AffinePoint> GetPointFromX(
      const BaseField& x) {
    TF_ASSIGN_OR_RETURN(BaseField y, GetYFromX(x));
    return AffinePoint(x, y);
  }

  // Returns the y-coordinate corresponding to the given x-coordinate if x lies
  // on the curve.
  constexpr static absl::StatusOr<BaseField> GetYFromX(const BaseField& x) {
    BaseField right = x.Square() * x + Config::kB;
    if constexpr (!Config::kA.IsZero()) {
      right += Config::kA * x;
    }
    return right.SquareRoot();
  }
};

}  // namespace zkx::math

#endif  // ZKX_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_SW_CURVE_H_
