#ifndef ZKX_MATH_BASE_SCALAR_MUL_H_
#define ZKX_MATH_BASE_SCALAR_MUL_H_

#include <stddef.h>

#include "zkx/math/base/big_int.h"
#include "zkx/math/base/bit_iterator.h"

namespace zkx::math {

template <typename T, size_t N>
[[nodiscard]] constexpr T ScalarMul(const T& value, const BigInt<N>& scalar) {
  T ret = T::Zero();
  auto it = BitIteratorBE<BigInt<N>>::begin(&scalar, true);
  auto end = BitIteratorBE<BigInt<N>>::end(&scalar);
  while (it != end) {
    ret = ret.Double();
    if (*it) {
      ret += value;
    }
    ++it;
  }
  return ret;
}

}  // namespace zkx::math

#endif  // ZKX_MATH_BASE_SCALAR_MUL_H_
