#ifndef ZKX_MATH_BASE_ENDIAN_UTILS_H_
#define ZKX_MATH_BASE_ENDIAN_UTILS_H_

#include <stddef.h>

#include "absl/base/internal/endian.h"

#if ABSL_IS_LITTLE_ENDIAN
#define FOR_FROM_BIGGEST(idx, start, end) \
  for (size_t idx = end - 1; idx != start - 1; --idx)

#define FOR_FROM_SMALLEST(idx, start, end) \
  for (size_t idx = start; idx < end; ++idx)

#define FOR_FROM_SECOND_SMALLEST(idx, start, end) \
  for (size_t idx = start + 1; idx < end; ++idx)

#else  // ABSL_IS_LITTLE_ENDIAN
#define FOR_FROM_BIGGEST(idx, start, end) \
  for (size_t idx = start; idx < end; ++idx)

#define FOR_FROM_SMALLEST(idx, start, end) \
  for (size_t idx = end - 1; idx != start - 1; --idx)

#define FOR_FROM_SECOND_SMALLEST(idx, start, end) \
  for (size_t idx = end - 2; idx != start - 1; --idx)

#endif

#endif  // ZKX_MATH_BASE_ENDIAN_UTILS_H_
