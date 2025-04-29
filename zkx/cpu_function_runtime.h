/* Copyright 2018 The OpenXLA Authors.

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

#ifndef ZKX_CPU_FUNCTION_RUNTIME_H_
#define ZKX_CPU_FUNCTION_RUNTIME_H_

#include <stddef.h>

#include "zkx/backends/cpu/alignment.h"

namespace zkx::cpu_function_runtime {

// Align to 64-bytes, to mimic tsl::Allocator::kAllocatorAlignment.
inline constexpr size_t Align() { return cpu::Align(); }

// The minimum alignment of buffers passed to ZKX:CPU.
inline constexpr size_t MinAlign() { return cpu::MinAlign(); }

}  // namespace zkx::cpu_function_runtime

#endif  // ZKX_CPU_FUNCTION_RUNTIME_H_
