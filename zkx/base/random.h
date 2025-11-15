#ifndef ZKX_BASE_RANDOM_H_
#define ZKX_BASE_RANDOM_H_

#include "absl/algorithm/container.h"
#include "absl/random/random.h"

namespace zkx::base {

absl::BitGen& GetAbslBitGen();

template <typename T>
T Uniform() {
  return absl::Uniform<T>(GetAbslBitGen());
}

template <typename T>
T Uniform(T low, T high) {
  return absl::Uniform<T>(GetAbslBitGen(), low, high);
}

template <typename Iterator>
void Shuffle(Iterator first, Iterator last) {
  absl::c_shuffle(absl::MakeSpan(first, last), GetAbslBitGen());
}

template <typename Container>
void Shuffle(Container& container) {
  absl::c_shuffle(container, GetAbslBitGen());
}

}  // namespace zkx::base

#endif  // ZKX_BASE_RANDOM_H_
