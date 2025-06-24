#ifndef ZKX_MATH_BASE_POW_H_
#define ZKX_MATH_BASE_POW_H_

#include <stddef.h>

#include "zkx/math/base/big_int.h"
#include "zkx/math/base/bit_iterator.h"

namespace zkx::math {

template <typename T, size_t N>
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

}  // namespace zkx::math

#endif  // ZKX_MATH_BASE_POW_H_
