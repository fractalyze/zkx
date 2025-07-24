#ifndef ZKX_MATH_BASE_EXTENSION_FIELD_H_
#define ZKX_MATH_BASE_EXTENSION_FIELD_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <ostream>
#include <string>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/strings/substitute.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/json/json_serde.h"
#include "zkx/base/strings/string_util.h"
#include "zkx/base/types/always_false.h"
#include "zkx/math/base/finite_field.h"
#include "zkx/math/base/pow.h"

namespace zkx {
namespace math {

template <typename _Config>
class ExtensionField : public FiniteField<ExtensionField<_Config>> {
 public:
  using Config = _Config;
  using BaseField = typename Config::BaseField;
  using BasePrimeField = typename Config::BasePrimeField;

  constexpr static uint32_t N = Config::kDegreeOverBaseField;
  constexpr static size_t kBitWidth = N * BaseField::kBitWidth;
  constexpr static size_t kByteWidth = N * BaseField::kByteWidth;

  constexpr ExtensionField() {
    for (size_t i = 0; i < N; ++i) {
      values_[i] = BaseField::Zero();
    }
  }

  template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
  constexpr ExtensionField(T value) {
    if (value >= 0) {
      *this = ExtensionField({value});
    } else {
      *this = -ExtensionField({-value});
    }
  }

  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr ExtensionField(T value) : ExtensionField({BigInt<N>(value)}) {}

  constexpr ExtensionField(std::initializer_list<BaseField> values) {
    DCHECK_LE(values.size(), N);
    auto it = values.begin();
    for (size_t i = 0; i < values.size(); ++i, ++it) {
      values_[i] = *it;
    }
    for (size_t i = values.size(); i < N; ++i) {
      values_[i] = BaseField::Zero();
    }
  }
  constexpr ExtensionField(const std::array<BaseField, N>& values)
      : values_(values) {}

  constexpr static uint32_t ExtensionDegree() {
    return N * BaseField::ExtensionDegree();
  }

  constexpr static ExtensionField Zero() { return ExtensionField(); }

  constexpr static ExtensionField One() {
    return ExtensionField({BaseField::One()});
  }

  constexpr static ExtensionField Random() {
    ExtensionField ret;
    for (size_t i = 0; i < std::size(ret.values_); ++i) {
      ret[i] = BaseField::Random();
    }
    return ret;
  }

  constexpr bool IsZero() const {
    for (size_t i = 0; i < std::size(values_); ++i) {
      if (!values_[i].IsZero()) return false;
    }
    return true;
  }

  constexpr bool IsOne() const {
    for (size_t i = 1; i < std::size(values_); ++i) {
      if (!values_[i].IsZero()) return false;
    }
    return values_[0].IsOne();
  }

  // See
  // https://github.com/Consensys/gnark-crypto/blob/43897fd/field/generator/internal/templates/extensions/e2.go.tmpl#L29-L37
  constexpr bool LexicographicallyLargest() const {
    for (size_t i = N - 1; i != SIZE_MAX; --i) {
      if (!values_[i].IsZero()) {
        return values_[i].LexicographicallyLargest();
      }
    }
    return false;
  }

  constexpr ExtensionField operator+(const ExtensionField& other) const {
    ExtensionField ret;
    for (size_t i = 0; i < std::size(values_); ++i) {
      ret[i] = values_[i] + other[i];
    }
    return ret;
  }

  constexpr ExtensionField& operator+=(const ExtensionField& other) {
    for (size_t i = 0; i < std::size(values_); ++i) {
      values_[i] += other[i];
    }
    return *this;
  }

  constexpr ExtensionField Double() const {
    ExtensionField ret;
    for (size_t i = 0; i < std::size(values_); ++i) {
      ret[i] = values_[i].Double();
    }
    return ret;
  }

  constexpr ExtensionField operator-(const ExtensionField& other) const {
    ExtensionField ret;
    for (size_t i = 0; i < std::size(values_); ++i) {
      ret[i] = values_[i] - other[i];
    }
    return ret;
  }

  constexpr ExtensionField& operator-=(const ExtensionField& other) {
    for (size_t i = 0; i < std::size(values_); ++i) {
      values_[i] -= other[i];
    }
    return *this;
  }

  constexpr ExtensionField operator-() const {
    ExtensionField ret;
    for (size_t i = 0; i < std::size(values_); ++i) {
      ret[i] = -values_[i];
    }
    return ret;
  }

  constexpr ExtensionField operator*(const ExtensionField& other) const {
    if constexpr (N == 2) {
      ExtensionField ret;
      DoMul2(*this, other, ret);
      return ret;
    } else {
      static_assert(base::AlwaysFalse<ExtensionField>, "Mul not implemented");
    }
    return ExtensionField::Zero();
  }

  constexpr ExtensionField operator*(const BaseField& other) const {
    ExtensionField ret;
    for (size_t i = 0; i < std::size(values_); ++i) {
      ret[i] = values_[i] * other;
    }
    return ret;
  }

  constexpr ExtensionField& operator*=(const ExtensionField& other) {
    if constexpr (N == 2) {
      DoMul2(*this, other, *this);
      return *this;
    } else {
      static_assert(base::AlwaysFalse<ExtensionField>, "Mul not implemented");
    }
    return *this = operator*(other);
  }

  constexpr ExtensionField& operator*=(const BaseField& other) {
    for (size_t i = 0; i < std::size(values_); ++i) {
      values_[i] *= other;
    }
    return *this;
  }

  constexpr ExtensionField Square() const {
    if constexpr (N == 2) {
      return DoSquare2(*this);
    }
    return operator*(*this);
  }

  template <size_t N>
  constexpr ExtensionField Pow(const BigInt<N>& exponent) const {
    return math::Pow(*this, exponent);
  }

  constexpr absl::StatusOr<ExtensionField> Inverse() const {
    if constexpr (N == 2) {
      return DoInverse2(*this);
    }
    return absl::UnimplementedError(
        absl::Substitute("Inverse not implemented for $0", N));
  }

  constexpr BaseField& operator[](size_t i) {
    DCHECK_LT(i, N);
    return values_[i];
  }
  constexpr const BaseField& operator[](size_t i) const {
    DCHECK_LT(i, N);
    return values_[i];
  }

  constexpr bool operator==(const ExtensionField& other) const {
    for (size_t i = 0; i < std::size(values_); ++i) {
      if (values_[i] != other[i]) return false;
    }
    return true;
  }
  constexpr bool operator!=(const ExtensionField& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return base::StrJoin(values_, [](std::ostream& os, const BaseField& value) {
      os << value.ToString();
    });
  }
  std::string ToHexString(bool pad_zero = false) const {
    return base::StrJoin(values_,
                         [pad_zero](std::ostream& os, const BaseField& value) {
                           os << value.ToHexString(pad_zero);
                         });
  }

 private:
  friend class base::Serde<ExtensionField>;
  friend class base::JsonSerde<ExtensionField>;

  constexpr static void DoMul2(const ExtensionField& a, const ExtensionField& b,
                               ExtensionField& c) {
    // clang-format off
    // (a[0], a[1]) * (b[0], b[1])
    //   = (a[0] + a[1] * x) * (b[0] + b[1] * x)
    //   = a[0] * b[0] + (a[0] * b[1] + a[1] * b[0]) * x + a[1] * b[1] * x²
    //   = a[0] * b[0] + a[1] * b[1] * x² + (a[0] * b[1] + a[1] * b[0]) * x
    //   = a[0] * b[0] + a[1] * b[1] * q + (a[0] * b[1] + a[1] * b[0]) * x
    //   = (a[0] * b[0] + a[1] * b[1] * q, a[0] * b[1] + a[1] * b[0])
    // where q is `Config::kNonResidue`.
    // clang-format on
    if constexpr (ExtensionDegree() == 2) {
      BaseField c0 = a[0] * b[0] + a[1] * b[1] * Config::kNonResidue;
      c[1] = a[0] * b[1] + a[1] * b[0];
      c[0] = c0;
    } else {
      // See https://www.math.u-bordeaux.fr/~damienrobert/csi/book/book.pdf
      // Karatsuba multiplication;
      // Guide to Pairing-based cryptography, Algorithm 5.16.
      // v0 = a[0] * b[0]
      BaseField v0 = a[0] * b[0];
      // v1 = a[1] * b[1]
      BaseField v1 = a[1] * b[1];

      // c[1] = (a[0] + a[1]) * (b[0] + b[1]) - v0 - v1
      // c[1] = a[0] * b[1] + a[1] * b[0]
      c[1] = (a[0] + a[1]) * (b[0] + b[1]) - v0 - v1;
      // c[0] = a[0] * b[0] + a[1] * b[1] * q
      c[0] = v0 + v1 * Config::kNonResidue;
    }
  }

  constexpr static ExtensionField DoSquare2(const ExtensionField& a) {
    // (a[0], a[1])² = (a[0] + a[1] * x)²
    //               = a[0]² + 2 * a[0] * a[1] * x + a[1]² * x²
    //               = a[0]² + a[1]² * x² + 2 * a[0] * a[1] * x
    //               = a[0]² + a[1]² * q + 2 * a[0] * a[1] * x
    //               = (a[0]² + a[1]² * q, 2 * a[0] * a[1])
    // where q is `Config::kNonResidue`.
    // When q = -1, we can re-use intermediate additions to improve performance.

    // v0 = a[0] - a[1]
    BaseField v0 = a[0] - a[1];
    // v1 = a[0] * a[1]
    BaseField v1 = a[0] * a[1];
    if constexpr (Config::kNonResidue == BaseField(-1)) {
      // When the non-residue is -1, we save 2 intermediate additions,
      // and use one fewer intermediate variable
      // (a[0]² - a[1]², 2 * a[0] * a[1])
      return {v0 * (a[0] + a[1]), v1.Double()};
    } else {
      // v2 = a[0] - a[1] * q
      BaseField v2 = a[0] - a[1] * Config::kNonResidue;

      // v0 = (v0 * v2)
      //    = (a[0] - a[1]) * (a[0] - a[1] * q)
      //    = a[0]² - a[0] * a[1] * q - a[0] * a[1] + a[1]² * q
      //    = a[0]² - (q + 1) * a[0] * a[1] + a[1]² * q
      //    = a[0]² + a[1]² * q - (q + 1) * a[0] * a[1]
      v0 *= v2;

      // clang-format off
      // a[0] = v0 + (q + 1) * a[0] * a[1]
      //      = a[0]² + a[1]² * q - (q + 1) * a[0] * a[1] + (q + 1) * a[0] * a[1]
      //      = a[0]² + a[1]² * q
      // clang-format on
      // a[1] = 2 * c0 * c1
      return {v0 + (Config::kNonResidue + 1) * v1, v1.Double()};
    }
  }

  constexpr static absl::StatusOr<ExtensionField> DoInverse2(
      const ExtensionField& a) {
    // See https://www.math.u-bordeaux.fr/~damienrobert/csi/book/book.pdf
    // Guide to Pairing-based Cryptography, Algorithm 5.19.
    // v1 = a[1]²
    BaseField v1 = a[1].Square();
    // v0 = a[0]² - q * v1
    BaseField v0 = a[0].Square() - v1 * Config::kNonResidue;

    TF_ASSIGN_OR_RETURN(BaseField v0_inv, v0.Inverse());
    return ExtensionField{a[0] * v0_inv, -a[1] * v0_inv};
  }

  std::array<BaseField, N> values_;
};

template <typename Config>
std::ostream& operator<<(std::ostream& os, const ExtensionField<Config>& ef) {
  return os << ef.ToHexString(true);
}

}  // namespace math

namespace base {

template <typename Config>
class Serde<math::ExtensionField<Config>> {
 public:
  constexpr static size_t N = math::ExtensionField<Config>::N;

  static absl::Status WriteTo(const math::ExtensionField<Config>& ext_field,
                              Buffer* buffer, Endian) {
    return buffer->Write(ext_field.values_);
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               math::ExtensionField<Config>* ext_field,
                               Endian) {
    for (size_t i = 0; i < N; ++i) {
      TF_RETURN_IF_ERROR(buffer.Read(&ext_field->values_[i]));
    }
    return absl::OkStatus();
  }

  static size_t EstimateSize(const math::ExtensionField<Config>& ext_field) {
    return base::EstimateSize(ext_field.values_);
  }
};

template <typename Config>
class JsonSerde<math::ExtensionField<Config>> {
 public:
  using BaseField = typename math::ExtensionField<Config>::BaseField;
  constexpr static size_t N = math::ExtensionField<Config>::N;

  template <typename Allocator>
  static rapidjson::Value From(const math::ExtensionField<Config>& value,
                               Allocator& allocator) {
    return JsonSerde<std::array<BaseField, N>>::From(value.values_, allocator);
  }

  static absl::StatusOr<math::ExtensionField<Config>> To(
      const rapidjson::Value& json_value, std::string_view key) {
    absl::StatusOr<std::array<BaseField, N>> values =
        JsonSerde<std::array<BaseField, N>>::To(json_value, key);
    if (values.ok()) {
      return math::ExtensionField<Config>(values.value());
    }
    return values.status();
  }
};

}  // namespace base
}  // namespace zkx

#endif  // ZKX_MATH_BASE_EXTENSION_FIELD_H_
