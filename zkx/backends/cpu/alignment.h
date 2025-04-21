/* Copyright 2024 The OpenXLA Authors.

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

#ifndef ZKX_BACKENDS_CPU_ALIGNMENT_H_
#define ZKX_BACKENDS_CPU_ALIGNMENT_H_

#include <cstddef>

#include "Eigen/Core"  // IWYU pragma: keep

namespace zkx::cpu {

// The minimum alignment of buffers passed to ZKX:CPU.
//
// ZKX:CPU emits code that assumes that all buffers passed to it are aligned to
// this boundary, and passing buffers with smaller alignment might lead to
// crashes at run time (illegal instruction loading from unaligned memory). We
// use the same alignment as Eigen, which is consistent with TensorFlow
// behavior.
inline constexpr size_t MinAlign() { return EIGEN_MAX_ALIGN_BYTES; }

// Align to 64-bytes, to mimic tsl::Allocator::kAllocatorAlignment.
//
// Preferred ZKX:CPU alignment for buffers. ZKX:CPU itself aligns intermediate
// buffers to this boundary (slices inside the preallocated temp buffer).
inline constexpr size_t Align() { return 64; }

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_ALIGNMENT_H_
