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

#include "zkx/math/poly/bit_reverse.h"

#include "absl/base/config.h"

namespace zkx::math {

uint64_t BitReverse64(uint64_t n) {
#if defined(__clang__) && ABSL_HAVE_BUILTIN(__builtin_convertvector)
  return __builtin_bitreverse64(n);
#else
  size_t count = 63;
  uint64_t rev = n;
  while ((n >>= 1) > 0) {
    rev <<= 1;
    rev |= n & 1;
    --count;
  }
  rev <<= count;
  return rev;
#endif
}

}  // namespace zkx::math
