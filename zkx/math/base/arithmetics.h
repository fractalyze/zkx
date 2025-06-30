#ifndef ZKX_MATH_BASE_ARITHMETICS_H_
#define ZKX_MATH_BASE_ARITHMETICS_H_

#include <stdint.h>

#include "absl/numeric/int128.h"

namespace zkx::math::internal {

template <typename T>
struct AddResult {
  T value{0};
  T carry{0};

  constexpr AddResult() = default;

  constexpr bool operator==(const AddResult& other) const {
    return value == other.value && carry == other.carry;
  }
  constexpr bool operator!=(const AddResult& other) const {
    return !operator==(other);
  }
};

template <typename T>
struct SubResult {
  T value{0};
  T borrow{0};

  constexpr SubResult() = default;

  constexpr bool operator==(const SubResult& other) const {
    return value == other.value && borrow == other.borrow;
  }
  constexpr bool operator!=(const SubResult& other) const {
    return !operator==(other);
  }
};

template <typename T>
struct MulResult {
  T hi{0};
  T lo{0};

  constexpr MulResult() = default;

  constexpr bool operator==(const MulResult& other) const {
    return hi == other.hi && lo == other.lo;
  }
  constexpr bool operator!=(const MulResult& other) const {
    return !operator==(other);
  }
};

template <typename T>
struct DivResult {
  T quotient{0};
  T remainder{0};

  constexpr DivResult() = default;

  constexpr bool operator==(const DivResult& other) const {
    return quotient == other.quotient && remainder == other.remainder;
  }
  constexpr bool operator!=(const DivResult& other) const {
    return !operator==(other);
  }
};

// Calculates a + b + carry.
constexpr AddResult<uint64_t> AddWithCarry(uint64_t a, uint64_t b,
                                           uint64_t carry = 0) {
  AddResult<uint64_t> result;
  absl::uint128 tmp =
      absl::uint128{a} + absl::uint128{b} + absl::uint128{carry};
  result.value = absl::Uint128Low64(tmp);
  result.carry = absl::Uint128High64(tmp);
  return result;
}

// Calculates a - b - borrow.
constexpr SubResult<uint64_t> SubWithBorrow(uint64_t a, uint64_t b,
                                            uint64_t borrow = 0) {
  SubResult<uint64_t> result;
  absl::uint128 tmp = (absl::uint128{1} << 64) + absl::uint128{a} -
                      absl::uint128{b} - absl::uint128{borrow};
  result.value = absl::Uint128Low64(tmp);
  result.borrow = absl::Uint128High64(tmp) == 0 ? 1 : 0;
  return result;
}

// Calculates a + b * c.
// NOTE(chokobole): absl::uint128 multiplication is not constexpr.
#if defined(__SIZEOF_INT128__)
constexpr MulResult<uint64_t> MulAddWithCarry(uint64_t a, uint64_t b,
                                              uint64_t c, uint64_t carry = 0) {
  __uint128_t tmp = static_cast<__uint128_t>(b) * c + a + carry;
  MulResult<uint64_t> result;
  result.lo = static_cast<uint64_t>(tmp);
  result.hi = static_cast<uint64_t>(tmp >> 64);
  return result;
}
#else
#error Define MulAddWithCarry
#endif

}  // namespace zkx::math::internal

#endif  // ZKX_MATH_BASE_ARITHMETICS_H_
