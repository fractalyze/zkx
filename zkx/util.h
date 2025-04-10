/* Copyright 2017 The OpenXLA Authors.

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

// Generally useful utility functions that are common to (not specific to any
// given part of) the ZKX code base.

#ifndef ZKX_UTIL_H_
#define ZKX_UTIL_H_

#include <stdint.h>

#include <limits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/macros.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

#include "xla/tsl/lib/math/math_util.h"
#include "zkx/status_macros.h"

namespace zkx {

// Ranks greater than 6 are very rare, so use InlinedVector<int64_t, 6> to store
// the bounds and indices. And for the rare cases of ranks greater than 6,
// the InlinedVector will just behave like an std::vector<> and allocate the
// memory to store its values.
inline constexpr int InlineRank() { return 6; }
using DimensionVector = absl::InlinedVector<int64_t, InlineRank()>;

// Imports the templated FloorOfRatio math function from the TensorFlow
// namespace, as it is very commonly used.
template <typename T>
constexpr T FloorOfRatio(T dividend, T divisor) {
  return tsl::MathUtil::FloorOfRatio<T>(dividend, divisor);
}

// Imports the templated CeilOfRatio math function from the TensorFlow
// namespace, as it is very commonly used.
template <typename T>
constexpr T CeilOfRatio(T dividend, T divisor) {
  return tsl::MathUtil::CeilOfRatio<T>(dividend, divisor);
}

// Rounds the value up to a multiple of the divisor by first calling CeilOfRatio
// then multiplying by the divisor. For example: RoundUpTo(13, 8) => 16
template <typename T>
constexpr T RoundUpTo(T value, T divisor) {
  return CeilOfRatio(value, divisor) * divisor;
}

// Rounds the value down to a multiple of the divisor by first calling
// FloorOfRatio then multiplying by the divisor. For example:
// RoundDownTo(13, 8) => 8
template <typename T>
constexpr T RoundDownTo(T value, T divisor) {
  return FloorOfRatio(value, divisor) * divisor;
}

// Returns a mask with "width" number of least significant bits set.
template <typename T>
constexpr inline T LsbMask(int width) {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  ABSL_ASSERT(width >= 0);
  ABSL_ASSERT(width <= std::numeric_limits<T>::digits);
  return width == 0
             ? 0
             : static_cast<T>(-1) >> (std::numeric_limits<T>::digits - width);
}

template <typename Container>
int64_t PositionInContainer(const Container& container, int64_t value) {
  return std::distance(container.begin(), absl::c_find(container, value));
}

int64_t Product(absl::Span<const int64_t> xs);

// Returns the start indices of consecutive non-overlapping subsequences of `a`
// and `b` with the same product, i.e. `(i, j)` so
// • a = {a[0 = i_0], ..., a[i_1 - 1], a[i_1], ... , a[i_2 - 1], ...}
// • b = {b[0 = j_0], ..., b[j_1 - 1], b[j_1], ... , b[j_2 - 1], ...}
// • ∀ k . 0 <= k < CommonFactors(a, b).size - 1 =>
//         a[i_k] × a[i_k + 1] × ... × a[i_(k+1) - 1] =
//         b[j_k] × b[j_k + 1] × ... × b[j_(k+1) - 1]
// where `CommonFactors(a, b)[CommonFactors(a, b).size - 1] = (a.size, b.size)`
//
// If input and output are the same, return {(0, 0), {1, 1}, ... {a.size,
// b.size}}, otherwise if the given shapes have non-zero size, returns the
// bounds of the shortest possible such subsequences; else, returns `{(0, 0),
// (a.size, b.size)}`.
absl::InlinedVector<std::pair<int64_t, int64_t>, 8> CommonFactors(
    absl::Span<const int64_t> a, absl::Span<const int64_t> b);

// Removes illegal characters from filenames.
std::string SanitizeFileName(std::string file_name);

template <typename T>
std::vector<T> SpanToVector(absl::Span<const T> slice) {
  return std::vector<T>(slice.begin(), slice.end());
}

template <typename T>
absl::Status EraseElementFromVector(std::vector<T>* container, const T& value) {
  // absl::c_find returns a const_iterator which does not seem to work on
  // gcc 4.8.4, and this breaks the ubuntu/xla_gpu build bot.
  auto it = std::find(container->begin(), container->end(), value);
  TF_RET_CHECK(it != container->end());
  container->erase(it);
  return absl::OkStatus();
}

// Returns a container with `sorted_ids_to_remove` elements removed.
template <typename T>
static T RemoveElements(absl::Span<const int64_t> sorted_ids_to_remove,
                        const T& container) {
  T result;
  auto id_to_remove = sorted_ids_to_remove.begin();
  for (size_t i = 0; i < container.size(); ++i) {
    if (id_to_remove != sorted_ids_to_remove.end() && *id_to_remove == i) {
      ++id_to_remove;
      continue;
    }
    result.push_back(container[i]);
  }
  return result;
}

}  // namespace zkx

#endif  // ZKX_UTIL_H_s
