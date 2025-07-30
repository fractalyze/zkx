#ifndef ZKX_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
#define ZKX_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_

#include <string>
#include <type_traits>

#include "absl/strings/substitute.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/json/json_serde.h"
#include "zkx/base/logging.h"
#include "zkx/math/base/finite_field_traits.h"
#include "zkx/math/base/scalar_mul.h"
#include "zkx/math/geometry/curve_type.h"
#include "zkx/math/geometry/point_declarations.h"

namespace zkx {
namespace math {

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

  constexpr static absl::StatusOr<AffinePoint> CreateFromX(const BaseField& x) {
    return Curve::GetPointFromX(x);
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
      return ScalarMul(ToJacobian(), v.MontReduce().value());
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

enum class AffinePointSerdeMode {
  kNone,
  // For Gnark's default format (i.e., data saved using Gnark's WriteTo())
  kGnarkDefault,
  // For Gnark's raw format (i.e., data saved using Gnark's WriteRawTo())
  kGnarkRaw,
};

// See
// https://github.com/Consensys/gnark-crypto/blob/43897fd/ecc/bn254/marshal.go#L21-L31
enum class GnarkCompressedFlag {
  kMask = 0b11 << 6,
  kUncompressed = 0b00 << 6,
  kCompressedSmallest = 0b10 << 6,
  kCompressedLargest = 0b11 << 6,
  kCompressedInfinity = 0b01 << 6,
};

}  // namespace math

namespace base {

template <typename Curve>
class Serde<math::AffinePoint<
    Curve,
    std::enable_if_t<Curve::kType == math::CurveType::kShortWeierstrass>>> {
 public:
  using BaseField = typename math::AffinePoint<Curve>::BaseField;
  using BasePrimeField =
      typename math::FiniteFieldTraits<BaseField>::BasePrimeField;

  static math::AffinePointSerdeMode s_mode;

  static absl::Status WriteTo(const math::AffinePoint<Curve>& point,
                              Buffer* buffer, Endian endian) {
    switch (s_mode) {
      case math::AffinePointSerdeMode::kNone:
        return buffer->WriteMany(point.x(), point.y());
      case math::AffinePointSerdeMode::kGnarkDefault:
      case math::AffinePointSerdeMode::kGnarkRaw: {
        if (endian != Endian::kBig) {
          return absl::InvalidArgumentError(
              "Invalid endian: GnarkDefault and GnarkRaw modes require "
              "BigEndian input");
        }
        if (Serde<BasePrimeField>::s_is_in_montgomery) {
          return absl::InvalidArgumentError(
              "Invalid format: Input must be in non-Montgomery form");
        }
        if (BasePrimeField::kBitWidth - BasePrimeField::kModulusBits < 2) {
          return absl::InvalidArgumentError(
              "Invalid format: BasePrimeField::kBitWidth - "
              "BasePrimeField::kModulusBits must be at least 2");
        }

        // See
        // https://github.com/Consensys/gnark-crypto/blob/43897fd/ecc/bn254/marshal.go#L790-L822.
        math::GnarkCompressedFlag flag;
        size_t old_buffer_offset = buffer->buffer_offset();
        if (point.IsZero()) {
          if (s_mode == math::AffinePointSerdeMode::kGnarkDefault) {
            uint8_t zeros[BaseField::kByteWidth] = {0};
            TF_RETURN_IF_ERROR(buffer->Write(zeros));
            flag = math::GnarkCompressedFlag::kCompressedInfinity;
          } else {
            uint8_t zeros[BaseField::kByteWidth * 2] = {0};
            TF_RETURN_IF_ERROR(buffer->Write(zeros));
            flag = math::GnarkCompressedFlag::kUncompressed;
          }
        } else {
          if (s_mode == math::AffinePointSerdeMode::kGnarkDefault) {
            if (point.y().LexicographicallyLargest()) {
              flag = math::GnarkCompressedFlag::kCompressedLargest;
            } else {
              flag = math::GnarkCompressedFlag::kCompressedSmallest;
            }
          } else {
            flag = math::GnarkCompressedFlag::kUncompressed;
          }
          if constexpr (BaseField::ExtensionDegree() == 1) {
            TF_RETURN_IF_ERROR(buffer->Write(point.x()));
            if (s_mode != math::AffinePointSerdeMode::kGnarkDefault) {
              TF_RETURN_IF_ERROR(buffer->Write(point.y()));
            }
          } else if constexpr (BaseField::ExtensionDegree() == 2) {
            TF_RETURN_IF_ERROR(buffer->Write(point.x()[1]));
            TF_RETURN_IF_ERROR(buffer->Write(point.x()[0]));
            if (s_mode != math::AffinePointSerdeMode::kGnarkDefault) {
              TF_RETURN_IF_ERROR(buffer->Write(point.y()[1]));
              TF_RETURN_IF_ERROR(buffer->Write(point.y()[0]));
            }
          } else {
            return absl::InvalidArgumentError(
                "Invalid extension degree: GnarkDefault and GnarkRaw modes "
                "only support extension degree 1 or 2");
          }
        }
        reinterpret_cast<uint8_t*>(buffer->buffer())[old_buffer_offset] |=
            static_cast<uint8_t>(flag);
        return absl::OkStatus();
      }
    }
    return absl::InvalidArgumentError("Unsupported AffinePointSerdeMode value");
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               math::AffinePoint<Curve>* point, Endian endian) {
    switch (s_mode) {
      case math::AffinePointSerdeMode::kNone: {
        BaseField x, y;
        TF_RETURN_IF_ERROR(buffer.ReadMany(&x, &y));
        *point = math::AffinePoint<Curve>(x, y);
        return absl::OkStatus();
      }
      case math::AffinePointSerdeMode::kGnarkDefault:
      case math::AffinePointSerdeMode::kGnarkRaw: {
        if (endian != Endian::kBig) {
          return absl::InvalidArgumentError(
              "Invalid endian: GnarkDefault and GnarkRaw modes require "
              "BigEndian input");
        }
        if (Serde<BasePrimeField>::s_is_in_montgomery) {
          return absl::InvalidArgumentError(
              "Invalid format: Input must be in non-Montgomery form");
        }
        if (BasePrimeField::kBitWidth - BasePrimeField::kModulusBits < 2) {
          return absl::InvalidArgumentError(
              "Invalid format: BasePrimeField::kBitWidth - "
              "BasePrimeField::kModulusBits must be at least 2");
        }
        // See
        // https://github.com/Consensys/gnark-crypto/blob/43897fd/ecc/bn254/marshal.go#L790-L822.
        std::array<uint8_t, BaseField::kByteWidth> bytes;
        TF_RETURN_IF_ERROR(buffer.Read(&bytes));
        auto flag = static_cast<math::GnarkCompressedFlag>(
            bytes[0] & static_cast<uint64_t>(math::GnarkCompressedFlag::kMask));
        bytes[0] &= 0x3F;
        switch (flag) {
          case math::GnarkCompressedFlag::kUncompressed: {
            std::array<uint8_t, BaseField::kByteWidth> bytes2;
            TF_RETURN_IF_ERROR(buffer.Read(&bytes2));
            BaseField x, y;
            if constexpr (BaseField::ExtensionDegree() == 1) {
              x = BasePrimeField(
                  math::BigInt<BaseField::N>::FromBytesBE(bytes));
              y = BasePrimeField(
                  math::BigInt<BaseField::N>::FromBytesBE(bytes2));
            } else if constexpr (BaseField::ExtensionDegree() == 2) {
              x = {
                  BasePrimeField(math::BigInt<BasePrimeField::N>::FromBytesBE(
                      absl::Span<const uint8_t>(
                          &bytes[BasePrimeField::kByteWidth],
                          BasePrimeField::kByteWidth))),
                  BasePrimeField(math::BigInt<BasePrimeField::N>::FromBytesBE(
                      absl::Span<const uint8_t>(&bytes[0],
                                                BasePrimeField::kByteWidth))),
              };
              y = {
                  BasePrimeField(math::BigInt<BasePrimeField::N>::FromBytesBE(
                      absl::Span<const uint8_t>(
                          &bytes2[BasePrimeField::kByteWidth],
                          BasePrimeField::kByteWidth))),
                  BasePrimeField(math::BigInt<BasePrimeField::N>::FromBytesBE(
                      absl::Span<const uint8_t>(&bytes2[0],
                                                BasePrimeField::kByteWidth))),
              };
            } else {
              return absl::InvalidArgumentError(
                  "Invalid extension degree: GnarkDefault and GnarkRaw modes "
                  "only support extension degree 1 or 2");
            }
            *point = math::AffinePoint<Curve>(x, y);
            return absl::OkStatus();
          }
          case math::GnarkCompressedFlag::kCompressedInfinity: {
            *point = math::AffinePoint<Curve>::Zero();
            return absl::OkStatus();
          }
          case math::GnarkCompressedFlag::kCompressedSmallest:
          case math::GnarkCompressedFlag::kCompressedLargest: {
            BaseField x;
            if constexpr (BaseField::ExtensionDegree() == 1) {
              x = BasePrimeField(
                  math::BigInt<BaseField::N>::FromBytesBE(bytes));
            } else if constexpr (BaseField::ExtensionDegree() == 2) {
              x = {
                  BasePrimeField(math::BigInt<BasePrimeField::N>::FromBytesBE(
                      absl::Span<const uint8_t>(
                          &bytes[BasePrimeField::kByteWidth],
                          BasePrimeField::kByteWidth))),
                  BasePrimeField(math::BigInt<BasePrimeField::N>::FromBytesBE(
                      absl::Span<const uint8_t>(&bytes[0],
                                                BasePrimeField::kByteWidth))),
              };
            } else {
              return absl::InvalidArgumentError(
                  "Invalid extension degree: GnarkDefault and GnarkRaw modes "
                  "only support extension degree 1 or 2");
            }
            TF_ASSIGN_OR_RETURN(*point,
                                math::AffinePoint<Curve>::CreateFromX(x));
            if (point->y().LexicographicallyLargest()) {
              if (flag == math::GnarkCompressedFlag::kCompressedSmallest) {
                *point = -*point;
              }
            } else {
              if (flag == math::GnarkCompressedFlag::kCompressedLargest) {
                *point = -*point;
              }
            }
            return absl::OkStatus();
          }
        }
        return absl::InvalidArgumentError(
            "Unsupported GnarkCompressedFlag value");
      }
    }
    return absl::InvalidArgumentError("Unsupported AffinePointSerdeMode value");
  }

  static size_t EstimateSize(const math::AffinePoint<Curve>& point) {
    switch (s_mode) {
      case math::AffinePointSerdeMode::kNone:
        return base::EstimateSize(point.x(), point.y());
      case math::AffinePointSerdeMode::kGnarkDefault:
        return base::EstimateSize(point.x());
      case math::AffinePointSerdeMode::kGnarkRaw: {
        if (point.IsZero()) {
          return base::EstimateSize(point.x());
        } else {
          return base::EstimateSize(point.x(), point.y());
        }
      }
    }
    ABSL_UNREACHABLE();
    return 0;
  }
};

// static
template <typename Curve>
math::AffinePointSerdeMode Serde<math::AffinePoint<
    Curve, std::enable_if_t<Curve::kType ==
                            math::CurveType::kShortWeierstrass>>>::s_mode =
    math::AffinePointSerdeMode::kNone;

template <typename Curve>
class JsonSerde<math::AffinePoint<
    Curve,
    std::enable_if_t<Curve::kType == math::CurveType::kShortWeierstrass>>> {
 public:
  using Field = typename math::AffinePoint<Curve>::BaseField;

  template <typename Allocator>
  static rapidjson::Value From(const math::AffinePoint<Curve>& value,
                               Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "x", value.x(), allocator);
    AddJsonElement(object, "y", value.y(), allocator);
    return object;
  }

  static absl::StatusOr<math::AffinePoint<Curve>> To(
      const rapidjson::Value& json_value, std::string_view key) {
    TF_ASSIGN_OR_RETURN(Field x, ParseJsonElement<Field>(json_value, "x"));
    TF_ASSIGN_OR_RETURN(Field y, ParseJsonElement<Field>(json_value, "y"));
    return math::AffinePoint<Curve>(x, y);
  }
};

}  // namespace base
}  // namespace zkx

#endif  // ZKX_MATH_ELLIPTIC_CURVES_AFFINE_POINT_H_
