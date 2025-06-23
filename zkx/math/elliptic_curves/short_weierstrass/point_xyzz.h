#ifndef ZKX_MATH_ELLIPTIC_CURVES_POINT_XYZZ_H_
#define ZKX_MATH_ELLIPTIC_CURVES_POINT_XYZZ_H_

#include <string>
#include <type_traits>

#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/base/json/json_serde.h"
#include "zkx/base/template_util.h"
#include "zkx/math/base/batch_inverse.h"
#include "zkx/math/base/scalar_mul.h"
#include "zkx/math/geometry/curve_type.h"
#include "zkx/math/geometry/point_declarations.h"

namespace zkx {
namespace math {

template <typename _Curve>
class PointXyzz<_Curve,
                std::enable_if_t<_Curve::kType == CurveType::kShortWeierstrass>>
    final {
 public:
  using Curve = _Curve;
  using BaseField = typename Curve::BaseField;
  using ScalarField = typename Curve::ScalarField;

  using AffinePoint = math::AffinePoint<Curve>;
  using JacobianPoint = math::JacobianPoint<Curve>;

  constexpr static size_t kBitWidth = BaseField::kBitWidth * 4;

  constexpr PointXyzz()
      : PointXyzz(BaseField::One(), BaseField::One(), BaseField::Zero(),
                  BaseField::Zero()) {}
  // NOTE(chokobole): This method is provided for testing purposes.
  template <typename T, std::enable_if_t<
                            std::is_constructible_v<ScalarField, T>>* = nullptr>
  constexpr PointXyzz(T value) : PointXyzz(ScalarField(value)) {}
  constexpr PointXyzz(ScalarField value) {
    PointXyzz point = PointXyzz::Generator() * value;
    x_ = point.x_;
    y_ = point.y_;
    zz_ = point.zz_;
    zzz_ = point.zzz_;
  }
  constexpr PointXyzz(const BaseField& x, const BaseField& y,
                      const BaseField& zz, const BaseField& zzz)
      : x_(x), y_(y), zz_(zz), zzz_(zzz) {}

  constexpr static PointXyzz Zero() { return PointXyzz(); }

  constexpr static PointXyzz One() { return Generator(); }

  constexpr static PointXyzz Generator() {
    return {Curve::Config::kX, Curve::Config::kY, BaseField::One(),
            BaseField::One()};
  }

  constexpr static PointXyzz Random() {
    return ScalarField::Random() * Generator();
  }

  constexpr const BaseField& x() const { return x_; }
  constexpr const BaseField& y() const { return y_; }
  constexpr const BaseField& zz() const { return zz_; }
  constexpr const BaseField& zzz() const { return zzz_; }

  constexpr bool IsZero() const { return zz_.IsZero(); }
  constexpr bool IsOne() const {
    return x_ == Curve::Config::kX && y_ == Curve::Config::kY && zz_.IsOne() &&
           zzz_.IsOne();
  }

  constexpr bool operator==(const PointXyzz& other) const {
    if (IsZero()) {
      return other.IsZero();
    }

    if (other.IsZero()) {
      return false;
    }

    // The points (X, Y, ZZ, ZZZ) and (X', Y', ZZ', ZZZ')
    // are equal when (X * ZZ') = (X' * ZZ)
    // and (Y * Z'³) = (Y' * Z³).
    if (x_ * other.zz_ != other.x_ * zz_) {
      return false;
    } else {
      return y_ * other.zzz_ == other.y_ * zzz_;
    }
  }

  constexpr bool operator!=(const PointXyzz& other) const {
    return !operator==(other);
  }

  constexpr PointXyzz operator+(const PointXyzz& other) const {
    if (IsZero()) {
      return other;
    }

    if (other.IsZero()) {
      return *this;
    }

    PointXyzz ret;
    Add(*this, other, ret);
    return ret;
  }

  constexpr PointXyzz& operator+=(const PointXyzz& other) {
    if (IsZero()) {
      return *this = other;
    }

    if (other.IsZero()) {
      return *this;
    }

    Add(*this, other, *this);
    return *this;
  }

  constexpr PointXyzz operator+(const AffinePoint& other) const {
    if (IsZero()) {
      return other.ToXyzz();
    }

    if (other.IsZero()) {
      return *this;
    }

    PointXyzz ret;
    Add(*this, other, ret);
    return ret;
  }

  constexpr PointXyzz& operator+=(const AffinePoint& other) {
    if (IsZero()) {
      return *this = other.ToXyzz();
    }

    if (other.IsZero()) {
      return *this;
    }

    Add(*this, other, *this);
    return *this;
  }

  constexpr PointXyzz Double() const;

  constexpr PointXyzz operator-(const PointXyzz& other) const {
    return operator+(-other);
  }

  constexpr PointXyzz& operator-=(const PointXyzz& other) {
    return *this = operator-(other);
  }

  constexpr PointXyzz operator-(const AffinePoint& other) const {
    return operator+(-other);
  }

  constexpr PointXyzz& operator-=(const AffinePoint& other) {
    return *this = operator-(other);
  }

  constexpr PointXyzz operator-() const { return {x_, -y_, zz_, zzz_}; }

  constexpr PointXyzz operator*(const ScalarField& v) const {
    if constexpr (ScalarField::kUseMontgomery) {
      return ScalarMul(*this, v.MontReduce());
    } else {
      return ScalarMul(*this, v.value());
    }
  }

  constexpr PointXyzz& operator*=(const ScalarField& v) {
    return *this = operator*(v);
  }

  // The xyzz point X, Y, ZZ, ZZZ is represented in the affine
  // coordinates as X/ZZ, Y/ZZZ.
  constexpr absl::StatusOr<AffinePoint> ToAffine() const {
    if (IsZero()) {
      return AffinePoint::Zero();
    } else if (zz_.IsOne()) {
      return AffinePoint(x_, y_);
    } else {
      // NOTE(ashjeong): if `zzz_` is 0, `IsZero()` will also evaluate to true,
      // and this block will not be executed
      TF_ASSIGN_OR_RETURN(BaseField z_inv_cubic, zzz_.Inverse());
      BaseField z_inv_square = (z_inv_cubic * zz_).Square();
      return AffinePoint(x_ * z_inv_square, y_ * z_inv_cubic);
    }
  }

  // The xyzz point X, Y, ZZ, ZZZ is represented in the jacobian
  // coordinates as X*ZZZ²*ZZ, Y*ZZ³*ZZZ², ZZZ*ZZ.
  constexpr JacobianPoint ToJacobian() const {
    if (IsZero()) {
      return JacobianPoint::Zero();
    } else if (zz_.IsOne()) {
      return {x_, y_, BaseField::One()};
    } else {
      BaseField z = zz_ * zzz_;
      return {x_ * zzz_ * z, y_ * zz_ * z.Square(), z};
    }
  }

  template <typename XyzzContainer, typename AffineContainer>
  static absl::Status BatchToAffine(const XyzzContainer& point_xyzzs,
                                    AffineContainer* affine_points) {
    if constexpr (base::internal::has_resize_v<AffineContainer>) {
      affine_points->resize(std::size(point_xyzzs));
    } else {
      if (std::size(point_xyzzs) != std::size(*affine_points)) {
        return absl::InvalidArgumentError(absl::Substitute(
            "size do not match $0 vs $1", std::size(point_xyzzs),
            std::size(*affine_points)));
      }
    }
    std::vector<BaseField> zzz_inverses = base::Map(
        point_xyzzs, [](const PointXyzz& point) { return point.zzz_; });
    TF_RETURN_IF_ERROR(BatchInverse(zzz_inverses, &zzz_inverses));
    for (size_t i = 0; i < std::size(*affine_points); ++i) {
      const PointXyzz& point_xyzz = point_xyzzs[i];
      if (point_xyzz.zz_.IsZero()) {
        (*affine_points)[i] = AffinePoint::Zero();
      } else if (point_xyzz.zz_.IsOne()) {
        (*affine_points)[i] = {point_xyzz.x_, point_xyzz.y_};
      } else {
        const BaseField& z_inv_cubic = zzz_inverses[i];
        BaseField z_inv_square = (z_inv_cubic * point_xyzz.zz_).Square();
        (*affine_points)[i] = {point_xyzz.x_ * z_inv_square,
                               point_xyzz.y_ * z_inv_cubic};
      }
    }
    return absl::OkStatus();
  }

  std::string ToString() const {
    return absl::Substitute("($0, $1, $2, $3)", x_.ToString(), y_.ToString(),
                            zz_.ToString(), zzz_.ToString());
  }
  std::string ToHexString(bool pad_zero = false) const {
    return absl::Substitute("($0, $1, $2, $3)", x_.ToHexString(pad_zero),
                            y_.ToHexString(pad_zero), zz_.ToHexString(pad_zero),
                            zzz_.ToHexString(pad_zero));
  }

 private:
  constexpr static void Add(const PointXyzz& a, const PointXyzz& b,
                            PointXyzz& c);
  constexpr static void Add(const PointXyzz& a, const AffinePoint& b,
                            PointXyzz& c);

  BaseField x_;
  BaseField y_;
  BaseField zz_;
  BaseField zzz_;
};

}  // namespace math

namespace base {

template <typename Curve>
class Serde<math::PointXyzz<
    Curve,
    std::enable_if_t<Curve::kType == math::CurveType::kShortWeierstrass>>> {
 public:
  static absl::Status WriteTo(const math::PointXyzz<Curve>& point,
                              Buffer* buffer) {
    return buffer->WriteMany(point.x(), point.y(), point.zz(), point.zzz());
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               math::PointXyzz<Curve>* point) {
    using BaseField = typename math::PointXyzz<Curve>::BaseField;
    BaseField x, y, zz, zzz;
    TF_RETURN_IF_ERROR(buffer.ReadMany(&x, &y, &zz, &zzz));
    *point = math::PointXyzz<Curve>(x, y, zz, zzz);
    return absl::OkStatus();
  }

  static size_t EstimateSize(const math::PointXyzz<Curve>& point) {
    return base::EstimateSize(point.x(), point.y(), point.zz(), point.zzz());
  }
};

template <typename Curve>
class JsonSerde<math::PointXyzz<
    Curve,
    std::enable_if_t<Curve::kType == math::CurveType::kShortWeierstrass>>> {
 public:
  using Field = typename math::PointXyzz<Curve>::BaseField;

  template <typename Allocator>
  static rapidjson::Value From(const math::PointXyzz<Curve>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "x", value.x(), allocator);
    AddJsonElement(object, "y", value.y(), allocator);
    AddJsonElement(object, "zz", value.zz(), allocator);
    AddJsonElement(object, "zzz", value.zzz(), allocator);
    return object;
  }

  static absl::StatusOr<math::PointXyzz<Curve>> To(
      const rapidjson::Value& json_value, std::string_view key) {
    TF_ASSIGN_OR_RETURN(Field x, ParseJsonElement<Field>(json_value, "x"));
    TF_ASSIGN_OR_RETURN(Field y, ParseJsonElement<Field>(json_value, "y"));
    TF_ASSIGN_OR_RETURN(Field zz, ParseJsonElement<Field>(json_value, "zz"));
    TF_ASSIGN_OR_RETURN(Field zzz, ParseJsonElement<Field>(json_value, "zzz"));
    return math::PointXyzz<Curve>(x, y, zz, zzz);
  }
};

}  // namespace base
}  // namespace zkx

#include "zkx/math/elliptic_curves/short_weierstrass/point_xyzz_impl.h"

#endif  // ZKX_MATH_ELLIPTIC_CURVES_POINT_XYZZ_H_
