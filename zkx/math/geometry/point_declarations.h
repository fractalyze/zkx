#ifndef ZKX_MATH_CURVE_POINT_DECLARATIONS_H_
#define ZKX_MATH_CURVE_POINT_DECLARATIONS_H_

#include <ostream>

namespace zkx::math {

template <typename Curve, typename SFINAE = void>
class AffinePoint;

template <typename Curve, typename SFINAE = void>
class JacobianPoint;

template <typename Curve, typename SFINAE = void>
class PointXyzz;

template <typename ScalarField, typename Curve,
          std::enable_if_t<std::is_same_v<
              ScalarField, typename Curve::ScalarField>>* = nullptr>
auto operator*(const ScalarField& v, const AffinePoint<Curve>& point) {
  return point * v;
}

template <typename ScalarField, typename Curve,
          std::enable_if_t<std::is_same_v<
              ScalarField, typename Curve::ScalarField>>* = nullptr>
auto operator*(const ScalarField& v, const JacobianPoint<Curve>& point) {
  return point * v;
}

template <typename ScalarField, typename Curve,
          std::enable_if_t<std::is_same_v<
              ScalarField, typename Curve::ScalarField>>* = nullptr>
auto operator*(const ScalarField& v, const PointXyzz<Curve>& point) {
  return point * v;
}

template <typename Curve>
std::ostream& operator<<(std::ostream& os, const AffinePoint<Curve>& point) {
  return os << point.ToHexString(true);
}

template <typename Curve>
std::ostream& operator<<(std::ostream& os, const JacobianPoint<Curve>& point) {
  return os << point.ToHexString(true);
}

template <typename Curve>
std::ostream& operator<<(std::ostream& os, const PointXyzz<Curve>& point) {
  return os << point.ToHexString(true);
}

}  // namespace zkx::math

#endif  // ZKX_MATH_CURVE_POINT_DECLARATIONS_H_
