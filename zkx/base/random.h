#ifndef ZKX_BASE_RANDOM_H_
#define ZKX_BASE_RANDOM_H_

#include "absl/random/random.h"

namespace zkx::base {

absl::BitGen& GetAbslBitGen();

template <typename T>
T Uniform() {
  return absl::Uniform<T>(GetAbslBitGen());
}

}  // namespace zkx::base

#endif  // ZKX_BASE_RANDOM_H_
