/* Copyright 2025 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ZKX_MATH_POLY_BIT_REVERSE_H_
#define ZKX_MATH_POLY_BIT_REVERSE_H_

#include <utility>
#include <vector>

#include "zkx/base/bits.h"
#include "zkx/base/openmp_util.h"

namespace zkx::math {

uint64_t BitReverse64(uint64_t n);

// Reverses the `bit_len` least significant bits of `x`.
inline size_t BitReverse(size_t x, uint32_t bit_len) {
  return BitReverse64(x) >> (sizeof(size_t) * 8 - bit_len);
}

// Reverses a container in-place according to bit-reversed indices.
// Example: index 3 (011) is swapped with index 6 (110) if size == 8.
template <typename Container>
void BitReverseShuffleInPlace(Container& container) {
  size_t size = std::size(container);
  if (size <= 1) return;

  uint32_t log_len = static_cast<uint32_t>(base::Log2Ceiling(size));
  OMP_PARALLEL_FOR(size_t idx = 1; idx < size; ++idx) {
    size_t ridx = BitReverse(idx, log_len);
    if (idx < ridx) {
      std::swap(container.at(idx), container.at(ridx));
    }
  }
}

// Returns a new vector where elements are rearranged according to bit-reversed
// indices.
template <typename T>
std::vector<T> BitReverseShuffle(const std::vector<T>& input) {
  std::vector<T> result = input;
  BitReverseShuffleInPlace(result);
  return result;
}

}  // namespace zkx::math

#endif  // ZKX_MATH_POLY_BIT_REVERSE_H_
