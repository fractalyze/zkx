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

#include "absl/container/inlined_vector.h"

namespace zkx {

// Ranks greater than 6 are very rare, so use InlinedVector<int64_t, 6> to store
// the bounds and indices. And for the rare cases of ranks greater than 6,
// the InlinedVector will just behave like an std::vector<> and allocate the
// memory to store its values.
inline constexpr int InlineRank() { return 6; }
using DimensionVector = absl::InlinedVector<int64_t, InlineRank()>;

}  // namespace zkx

#endif  // ZKX_UTIL_H_s
