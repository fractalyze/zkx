#ifndef ZKX_MATH_BASE_PRIME_FIELD_H_
#define ZKX_MATH_BASE_PRIME_FIELD_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <ostream>
#include <string>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/status/statusor.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/json/json_serde.h"
#include "zkx/math/base/big_int.h"
#include "zkx/math/base/byinverter.h"
#include "zkx/math/base/finite_field.h"
#include "zkx/math/base/pow.h"

namespace zkx {
namespace math {

// This implements PrimeField on Montgomery domain.
template <typename _Config>
class PrimeField : public FiniteField<PrimeField<_Config>> {
 public:
  constexpr static bool kUseMontgomery = true;
  constexpr static bool kUseBigModulus = true;
  constexpr static size_t kModulusBits = _Config::kModulusBits;
  constexpr static size_t kLimbNums = (kModulusBits + 63) / 64;
  constexpr static size_t N = kLimbNums;
  constexpr static size_t kBitWidth = BigInt<N>::kBitWidth;
  constexpr static size_t kByteWidth = BigInt<N>::kByteWidth;

  using Config = _Config;

  constexpr static BYInverter<N> s_inverter_ =
      BYInverter<N>(Config::kModulus, Config::kRSquared);

  constexpr PrimeField() = default;
  template <typename T, std::enable_if_t<std::is_signed_v<T>>* = nullptr>
  constexpr PrimeField(T value) {
    if (value >= 0) {
      *this = PrimeField(BigInt<N>(value));
    } else {
      *this = -PrimeField(BigInt<N>(-value));
    }
  }
  template <typename T, std::enable_if_t<std::is_unsigned_v<T>>* = nullptr>
  constexpr PrimeField(T value) : PrimeField(BigInt<N>(value)) {}
  template <typename T>
  constexpr PrimeField(std::initializer_list<T> values)
      : PrimeField(BigInt<N>(values)) {}
  constexpr explicit PrimeField(const BigInt<N>& value) : value_(value) {
    DCHECK_LT(value_, Config::kModulus);
    operator*=(PrimeField::FromUnchecked(Config::kRSquared));
  }

  constexpr static uint32_t ExtensionDegree() { return 1; }

  constexpr static PrimeField Zero() { return PrimeField(); }

  constexpr static PrimeField One() {
    return PrimeField::FromUnchecked(Config::kOne);
  }

  constexpr static PrimeField Min() { return Zero(); }

  constexpr static PrimeField Max() {
    return PrimeField::FromUnchecked(Config::kModulus - 1);
  }

  constexpr static PrimeField Random() {
    return PrimeField::FromUnchecked(BigInt<N>::Random(Config::kModulus));
  }

  constexpr static PrimeField FromUnchecked(const BigInt<N>& value) {
    PrimeField ret;
    ret.value_ = value;
    return ret;
  }

  // Convert a decimal string to a PrimeField.
  static absl::StatusOr<PrimeField> FromDecString(std::string_view str) {
    TF_ASSIGN_OR_RETURN(BigInt<N> ret_value, BigInt<N>::FromDecString(str));
    return PrimeField(ret_value);
  }

  // Convert a hexadecimal string to a PrimeField.
  static absl::StatusOr<PrimeField> FromHexString(std::string_view str) {
    TF_ASSIGN_OR_RETURN(BigInt<N> ret_value, BigInt<N>::FromHexString(str));
    return PrimeField(ret_value);
  }

  constexpr const BigInt<N>& value() const { return value_; }

  constexpr bool IsZero() const { return value_.IsZero(); }

  constexpr bool IsOne() const { return value_ == Config::kOne; }

  constexpr PrimeField operator+(const PrimeField& other) const {
    BigInt<N> ret_value;
    bool carry = false;
    if constexpr (HasSpareBit()) {
      ret_value = value_ + other.value_;
    } else {
      carry = BigInt<N>::Add(value_, other.value_, ret_value);
    }
    Clamp(ret_value, carry);
    return PrimeField::FromUnchecked(ret_value);
  }

  constexpr PrimeField& operator+=(const PrimeField& other) {
    return *this = *this + other;
  }

  constexpr PrimeField Double() const {
    BigInt<N> ret_value;
    bool carry = false;
    if constexpr (HasSpareBit()) {
      ret_value = value_ << 1;
    } else {
      carry = BigInt<N>::ShiftLeft(value_, ret_value, 1);
    }
    Clamp(ret_value, carry);
    return PrimeField::FromUnchecked(ret_value);
  }

  constexpr PrimeField operator-(const PrimeField& other) const {
    BigInt<N> ret_value;
    if (other.value_ > value_) {
      static_assert(HasSpareBit());
      ret_value = value_ + Config::kModulus;
      ret_value -= other.value_;
    } else {
      ret_value = value_ - other.value_;
    }
    return PrimeField::FromUnchecked(ret_value);
  }

  constexpr PrimeField& operator-=(const PrimeField& other) {
    return *this = *this - other;
  }

  constexpr PrimeField operator-() const {
    if (IsZero()) return Zero();
    BigInt<N> ret_value = Config::kModulus;
    ret_value -= value_;
    return PrimeField::FromUnchecked(ret_value);
  }

  constexpr PrimeField operator*(const PrimeField& other) const {
    PrimeField ret;
    if constexpr (CanUseNoCarryMulOptimization()) {
      FastMul(*this, other, ret);
    } else {
      SlowMul(*this, other, ret);
    }
    return ret;
  }

  constexpr PrimeField& operator*=(const PrimeField& other) {
    if constexpr (CanUseNoCarryMulOptimization()) {
      FastMul(*this, other, *this);
    } else {
      SlowMul(*this, other, *this);
    }
    return *this;
  }

  constexpr PrimeField Square() const {
    PrimeField ret;
    FastSquare(*this, ret);
    return ret;
  }

  template <typename T>
  constexpr PrimeField Pow(const T& exponent) const {
    if constexpr (std::is_same_v<T, PrimeField>) {
      return math::Pow(*this, exponent.MontReduce());
    } else {
      return math::Pow(*this, BigInt<N>(exponent));
    }
  }

  constexpr absl::StatusOr<PrimeField> operator/(
      const PrimeField& other) const {
    TF_ASSIGN_OR_RETURN(PrimeField inv, other.Inverse());
    return operator*(inv);
  }

  constexpr absl::StatusOr<PrimeField> Inverse() const {
    PrimeField ret;
    if (!s_inverter_.Invert(value_, ret.value_)) {
      return absl::InvalidArgumentError("division by zero");
    }
    return ret;
  }

  constexpr uint64_t& operator[](size_t i) {
    DCHECK_LT(i, N);
    return value_[i];
  }
  constexpr const uint64_t operator[](size_t i) const {
    DCHECK_LT(i, N);
    return value_[i];
  }

  constexpr bool operator==(const PrimeField& other) const {
    return value_ == other.value_;
  }

  constexpr bool operator!=(const PrimeField& other) const {
    return !operator==(other);
  }

  constexpr bool operator<(const PrimeField& other) const {
    return MontReduce() < other.MontReduce();
  }

  constexpr bool operator>(const PrimeField& other) const {
    return MontReduce() > other.MontReduce();
  }

  constexpr bool operator<=(const PrimeField& other) const {
    return MontReduce() <= other.MontReduce();
  }

  constexpr bool operator>=(const PrimeField& other) const {
    return MontReduce() >= other.MontReduce();
  }

  BigInt<N> MontReduce() const {
    BigInt<N> ret = value_;
    for (size_t i = 0; i < N; ++i) {
      uint64_t k = ret[i] * Config::kNPrime;
      internal::MulResult<uint64_t> result =
          internal::MulAddWithCarry(ret[i], k, Config::kModulus[0], 0);
      for (size_t j = 1; j < N; ++j) {
        result = internal::MulAddWithCarry(ret[(j + i) % N], k,
                                           Config::kModulus[j], result.hi);
        ret[(j + i) % N] = result.lo;
      }
      ret[i] = result.hi;
    }
    return ret;
  }

  std::string ToString() const { return MontReduce().ToString(); }
  std::string ToHexString(bool pad_zero = false) const {
    return MontReduce().ToHexString(pad_zero);
  }

 private:
  // Does the modulus have a spare unused bit?
  //
  // This condition applies if
  // (a) `modulus[biggest_limb_idx] >> 63 == 0`
  constexpr static bool HasSpareBit() {
    uint64_t biggest_limb = Config::kModulus[N - 1];
    return biggest_limb >> 63 == 0;
  }

  // Can we use the no-carry optimization for multiplication
  // outlined [here](https://hackmd.io/@gnark/modular_multiplication)?
  //
  // This optimization applies if
  // (a) `modulus[biggest_limb_idx] < max(uint64_t) >> 1`, and
  // (b) the bits of the modulus are not all 1.
  constexpr static bool CanUseNoCarryMulOptimization() {
    uint64_t biggest_limb = Config::kModulus[N - 1];
    bool top_bit_is_zero = biggest_limb >> 63 == 0;
    bool all_remaining_bits_are_one =
        biggest_limb == std::numeric_limits<uint64_t>::max() >> 1;
    for (size_t i = 1; i < N - 1; ++i) {
      all_remaining_bits_are_one &=
          Config::kModulus[i] == std::numeric_limits<uint64_t>::max();
    }
    all_remaining_bits_are_one &=
        Config::kModulus[0] == std::numeric_limits<uint64_t>::max();
    return top_bit_is_zero && !all_remaining_bits_are_one;
  }

  constexpr static void Clamp(BigInt<N>& value,
                              [[maybe_unused]] bool carry = false) {
    bool needs_to_clamp = value >= Config::kModulus;
    if constexpr (!HasSpareBit()) {
      needs_to_clamp |= carry;
    }
    if (needs_to_clamp) {
      value -= Config::kModulus;
    }
  }

  constexpr static void FastMul(const PrimeField& a, const PrimeField& b,
                                PrimeField& c) {
    BigInt<N> ret;
    for (size_t i = 0; i < N; ++i) {
      internal::MulResult<uint64_t> result;
      result = internal::MulAddWithCarry(ret[0], a[0], b[i]);
      ret[0] = result.lo;

      uint64_t k = ret[0] * Config::kNPrime;
      internal::MulResult<uint64_t> result2;
      result2 = internal::MulAddWithCarry(ret[0], k, Config::kModulus[0]);

      for (size_t j = 1; j < N; ++j) {
        result = internal::MulAddWithCarry(ret[j], a[j], b[i], result.hi);
        ret[j] = result.lo;
        result2 = internal::MulAddWithCarry(ret[j], k, Config::kModulus[j],
                                            result2.hi);
        ret[j - 1] = result2.lo;
      }
      ret[N - 1] = result.hi + result2.hi;
    }
    Clamp(ret, 0);
    c.value_ = ret;
  }

  constexpr static void SlowMul(const PrimeField& a, const PrimeField& b,
                                PrimeField& c) {
    internal::MulResult<BigInt<N>> mul_result =
        BigInt<N>::Mul(a.value_, b.value_);
    BigInt<2 * N> mul;
    memcpy(&mul[0], &mul_result.lo[0], sizeof(uint64_t) * N);
    memcpy(&mul[N], &mul_result.hi[0], sizeof(uint64_t) * N);
    MontMulReduce(mul, c.value_);
  }

  constexpr static void FastSquare(const PrimeField& a, PrimeField& b) {
    BigInt<2 * N> ret;
    internal::MulResult<uint64_t> mul_result;
    for (size_t i = 0; i < N - 1; ++i) {
      for (size_t j = i + 1; j < N; ++j) {
        mul_result =
            internal::MulAddWithCarry(ret[i + j], a[i], a[j], mul_result.hi);
        ret[i + j] = mul_result.lo;
      }
      ret[i + N] = mul_result.hi;
      mul_result.hi = 0;
    }

    ret[2 * N - 1] = ret[2 * N - 2] >> 63;
    for (size_t i = 2; i < 2 * N - 1; ++i) {
      ret[2 * N - i] = (ret[2 * N - i] << 1) | (ret[2 * N - (i + 1)] >> 63);
    }
    ret[1] <<= 1;

    internal::AddResult<uint64_t> add_result;
    for (size_t i = 0; i < N; ++i) {
      mul_result =
          internal::MulAddWithCarry(ret[2 * i], a[i], a[i], mul_result.hi);
      ret[2 * i] = mul_result.lo;
      add_result = internal::AddWithCarry(ret[2 * i + 1], mul_result.hi);
      ret[2 * i + 1] = add_result.value;
      mul_result.hi = add_result.carry;
    }
    MontMulReduce(ret, b.value_);
  }

  constexpr static void MontMulReduce(BigInt<2 * N>& a, BigInt<N>& b) {
    internal::AddResult<uint64_t> add_result;
    for (size_t i = 0; i < N; ++i) {
      uint64_t tmp = a[i] * Config::kNPrime;
      internal::MulResult<uint64_t> mul_result;
      mul_result = internal::MulAddWithCarry(a[i], tmp, Config::kModulus[0],
                                             mul_result.hi);
      for (size_t j = 1; j < N; ++j) {
        mul_result = internal::MulAddWithCarry(
            a[i + j], tmp, Config::kModulus[j], mul_result.hi);
        a[i + j] = mul_result.lo;
      }
      add_result =
          internal::AddWithCarry(a[N + i], mul_result.hi, add_result.carry);
      a[N + i] = add_result.value;
    }
    memcpy(&b[0], &a[N], sizeof(uint64_t) * N);
    Clamp(b, add_result.carry);
  }

  BigInt<N> value_;
};

template <typename Config>
std::ostream& operator<<(std::ostream& os, const PrimeField<Config>& pf) {
  return os << pf.ToHexString(true);
}

template <typename T>
struct IsPrimeFieldImpl {
  constexpr static bool value = false;
};

template <typename Config>
struct IsPrimeFieldImpl<PrimeField<Config>> {
  constexpr static bool value = true;
};

template <typename T>
constexpr bool IsPrimeField = IsPrimeFieldImpl<T>::value;

}  // namespace math

namespace base {

template <typename Config>
class Serde<math::PrimeField<Config>> {
 public:
  constexpr static size_t N = math::PrimeField<Config>::N;

  static bool s_is_in_montgomery;

  static absl::Status WriteTo(const math::PrimeField<Config>& prime_field,
                              Buffer* buffer, Endian) {
    if (s_is_in_montgomery) {
      return buffer->Write(prime_field.value());
    } else {
      return buffer->Write(prime_field.MontReduce());
    }
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               math::PrimeField<Config>* prime_field, Endian) {
    math::BigInt<N> v;
    TF_RETURN_IF_ERROR(buffer.Read(&v));
    if (s_is_in_montgomery) {
      *prime_field = math::PrimeField<Config>::FromUnchecked(v);
    } else {
      *prime_field = math::PrimeField<Config>(v);
    }
    return absl::OkStatus();
  }

  static size_t EstimateSize(const math::PrimeField<Config>& prime_field) {
    return N * sizeof(uint64_t);
  }
};

// static
template <typename Config>
bool Serde<math::PrimeField<Config>>::s_is_in_montgomery = true;

template <typename Config>
class JsonSerde<math::PrimeField<Config>> {
 public:
  constexpr static size_t N = math::PrimeField<Config>::N;

  static bool s_is_in_montgomery;

  template <typename Allocator>
  static rapidjson::Value From(const math::PrimeField<Config>& value,
                               Allocator& allocator) {
    if (s_is_in_montgomery) {
      return JsonSerde<math::BigInt<N>>::From(value.value(), allocator);
    } else {
      return JsonSerde<math::BigInt<N>>::From(value.MontReduce(), allocator);
    }
  }

  static absl::StatusOr<math::PrimeField<Config>> To(
      const rapidjson::Value& json_value, std::string_view key) {
    TF_ASSIGN_OR_RETURN(math::BigInt<N> v,
                        JsonSerde<math::BigInt<N>>::To(json_value, key));

    if (s_is_in_montgomery) {
      return math::PrimeField<Config>::FromUnchecked(v);
    } else {
      return math::PrimeField<Config>(v);
    }
  }
};

// static
template <typename Config>
bool JsonSerde<math::PrimeField<Config>>::s_is_in_montgomery = true;

}  // namespace base
}  // namespace zkx

#endif  // ZKX_MATH_BASE_PRIME_FIELD_H_
