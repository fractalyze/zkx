#ifndef ZKX_MATH_FIELD_PRIME_FIELD_H_
#define ZKX_MATH_FIELD_PRIME_FIELD_H_

#include <stddef.h>

#include <ostream>

#include "absl/status/status.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/json/json_serde.h"
#include "zkx/math/base/big_int.h"
#include "zkx/math/field/finite_field_traits.h"

namespace zkx {
namespace math {

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

template <typename Config>
std::ostream& operator<<(std::ostream& os, const PrimeField<Config>& pf) {
  return os << pf.ToHexString(true);
}

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
      return buffer->Write(prime_field.MontReduce().value());
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
      return JsonSerde<math::BigInt<N>>::From(value.MontReduce().value(),
                                              allocator);
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

#endif  // ZKX_MATH_FIELD_PRIME_FIELD_H_
