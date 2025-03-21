/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PLATFORM_MEM_H_
#define XLA_TSL_PLATFORM_MEM_H_

#include <stddef.h>

namespace tsl::port {

// Aligned allocation/deallocation. `minimum_alignment` must be a power of 2
// and a multiple of sizeof(void*).
void* AlignedMalloc(size_t size, int minimum_alignment);
void AlignedFree(void* aligned_memory);
void AlignedSizedFree(void* aligned_memory, size_t alignment, size_t size);

// An allocator that allocates memory with the given minimum alignment.
template <class T, size_t minimum_alignment>
struct AlignedAllocator {
  using value_type = T;

  value_type* allocate(size_t n) {
    return static_cast<value_type*>(
        AlignedMalloc(n * sizeof(value_type), minimum_alignment));
  }

  void deallocate(value_type* p, size_t n) {
    return AlignedSizedFree(p, minimum_alignment, n);
  }
};

void* Malloc(size_t size);
void* Realloc(void* ptr, size_t size);
void Free(void* ptr);

}  // namespace tsl::port

#endif  // XLA_TSL_PLATFORM_MEM_H_
