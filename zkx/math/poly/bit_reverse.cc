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
