#ifndef ZKX_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
#define ZKX_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_

#include <string>
#include <type_traits>

#include "absl/strings/substitute.h"

#include "zkx/base/logging.h"
#include "zkx/math/base/scalar_mul.h"
#include "zkx/math/geometry/curve_type.h"
#include "zkx/math/geometry/point_declarations.h"

namespace zkx::math {

template <typename _Curve>
class AffinePoint<
    _Curve, std::enable_if_t<_Curve::kType == CurveType::kShortWeierstrass>>
    final {
 public:
  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;

  using JacobianPoint = math::JacobianPoint<Curve>;
  using PointXyzz = math::PointXyzz<Curve>;

  constexpr static size_t kBitWidth = BaseField::kBitWidth * 2;

  constexpr AffinePoint() : AffinePoint(BaseField::Zero(), BaseField::Zero()) {}
  // NOTE(chokobole): This method is provided for testing purposes.
  // For example, a user can create an affine point using this constructor:
  //
  //   Literal g1 = LiteralUtil::CreateR0<math::bn254::G1Affine>(1);
  template <typename T, std::enable_if_t<
                            std::is_constructible_v<ScalarField, T>>* = nullptr>
  constexpr AffinePoint(T value) : AffinePoint(ScalarField(value)) {}
  constexpr AffinePoint(ScalarField value) {
    AffinePoint point = *(AffinePoint::Generator() * value).ToAffine();
    x_ = point.x_;
    y_ = point.y_;
  }
  constexpr AffinePoint(const BaseField& x, const BaseField& y)
      : x_(x), y_(y) {}

  constexpr static AffinePoint Zero() { return AffinePoint(); }

  constexpr static AffinePoint One() { return Generator(); }

  constexpr static AffinePoint Generator() {
    return {Curve::Config::kX, Curve::Config::kY};
  }

  constexpr static AffinePoint Random() {
    return *JacobianPoint::Random().ToAffine();
  }

  constexpr const BaseField& x() const { return x_; }
  constexpr const BaseField& y() const { return y_; }

  constexpr bool IsZero() const { return x_.IsZero() && y_.IsZero(); }
  constexpr bool IsOne() const {
    return x_ == Curve::Config::kX && y_ == Curve::Config::kY;
  }

  constexpr bool operator==(const AffinePoint& other) const {
    return x_ == other.x_ && y_ == other.y_;
  }

  constexpr bool operator!=(const AffinePoint& other) const {
    return !operator==(other);
  }

  constexpr JacobianPoint operator+(const AffinePoint& other) const {
    return ToJacobian() + other;
  }
  constexpr JacobianPoint operator+(const JacobianPoint& other) const {
    return other + *this;
  }
  constexpr AffinePoint& operator+=(const AffinePoint& other) {
    LOG(FATAL) << "Invalid call to operator+=; this exists only to allow "
                  "compilation with reduction. See in_process_communicator.cc "
                  "for details";
    return *this;
  }
  constexpr JacobianPoint operator-(const AffinePoint& other) const {
    return ToJacobian() - other;
  }
  constexpr JacobianPoint operator-(const JacobianPoint& other) const {
    return -(other - *this);
  }
  constexpr AffinePoint operator-() const { return {x_, -y_}; }

  constexpr JacobianPoint operator*(const ScalarField& v) const {
    if constexpr (ScalarField::kUseMontgomery) {
      return ScalarMul(ToJacobian(), v.MontReduce());
    } else {
      return ScalarMul(ToJacobian(), v.value());
    }
  }

  constexpr JacobianPoint ToJacobian() const {
    if (IsZero()) return JacobianPoint::Zero();
    return {x_, y_, BaseField::One()};
  }

  constexpr PointXyzz ToXyzz() const {
    if (IsZero()) return PointXyzz::Zero();
    return {x_, y_, BaseField::One(), BaseField::One()};
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1)", x_.ToString(), y_.ToString());
  }
  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1)", x_.ToHexString(pad_zero),
                            y_.ToHexString(pad_zero));
  }

 private:
  BaseField x_;
  BaseField y_;
};

}  // namespace zkx::math

#endif  // ZKX_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
