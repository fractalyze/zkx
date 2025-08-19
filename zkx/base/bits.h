#ifndef ZKX_BASE_BITS_H_
#define ZKX_BASE_BITS_H_

#include <type_traits>

#include "absl/numeric/bits.h"

namespace zkx::base {

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
template <typename T>
constexpr inline int Log2Floor(T x) {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  return absl::bit_width(x) - 1;
}

// Return ceiling(log2(n)) for positive integer n.  Returns -1 iff n == 0.
template <typename T>
constexpr inline int Log2Ceiling(T x) {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  return x == 0 ? -1 : absl::bit_width(x - 1);
}

}  // namespace zkx::base

#endif  // ZKX_BASE_BITS_H_
