#ifndef ZKX_MATH_BASE_POW_H_
#define ZKX_MATH_BASE_POW_H_

#include <stddef.h>

#include <type_traits>

#include "zkx/math/base/big_int.h"
#include "zkx/math/base/bit_iterator.h"

namespace zkx::math {

template <typename T, size_t N,
          std::enable_if_t<!std::is_integral_v<T>>* = nullptr>
[[nodiscard]] constexpr T Pow(const T& value, const BigInt<N>& exponent) {
  T ret = T::One();
  auto it = BitIteratorBE<BigInt<N>>::begin(&exponent, true);
  auto end = BitIteratorBE<BigInt<N>>::end(&exponent);
  while (it != end) {
    ret = ret.Square();
    if (*it) {
      ret *= value;
    }
    ++it;
  }
  return ret;
}

template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
[[nodiscard]] constexpr T Pow(T value, T exponent) {
  T ret = 1;
  if (exponent < 0) {
    // This function is only called with non-negative exponents in the tests.
    // Add a contract (e.g., DCHECK) or proper handling for negative exponents.
    return 0;
  }

  T exp = exponent;
  while (exp > 0) {
    if (exp & 1) {
      ret *= value;
    }
    value *= value;
    exp >>= 1;
  }
  return ret;
}

}  // namespace zkx::math

#endif  // ZKX_MATH_BASE_POW_H_
