/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright 2025 The ZKX Authors.

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

#ifndef XLA_TSL_FRAMEWORK_ALLOCATOR_H_
#define XLA_TSL_FRAMEWORK_ALLOCATOR_H_

#include <stdint.h>

#include <optional>
#include <string>

namespace tsl {

// Runtime statistics collected by an allocator. Exactly the same as
// stream_executor::AllocatorStats, but independently defined to preserve the
// mutual independence of StreamExecutor and TensorFlow.
struct AllocatorStats {
  int64_t num_allocs;          // Number of allocations.
  int64_t bytes_in_use;        // Number of bytes in use.
  int64_t peak_bytes_in_use;   // The peak bytes in use.
  int64_t largest_alloc_size;  // The largest single allocation seen.

  // The upper limit of bytes of user allocatable device memory, if such a limit
  // is known.
  std::optional<int64_t> bytes_limit;

  // Stats for reserved memory usage.
  int64_t bytes_reserved;       // Number of bytes reserved.
  int64_t peak_bytes_reserved;  // The peak number of bytes reserved.
  // The upper limit on the number bytes of reservable memory,
  // if such a limit is known.
  std::optional<int64_t> bytes_reservable_limit;

  int64_t largest_free_block_bytes;  // Largest free block's size in heap.

  // Number of bytes of memory held by the allocator.  This may be higher than
  // bytes_in_use if the allocator holds a pool of memory (e.g. BFCAllocator).
  std::optional<int64_t> pool_bytes;
  std::optional<int64_t> peak_pool_bytes;

  AllocatorStats()
      : num_allocs(0),
        bytes_in_use(0),
        peak_bytes_in_use(0),
        largest_alloc_size(0),
        bytes_reserved(0),
        peak_bytes_reserved(0),
        largest_free_block_bytes(0) {}

  std::string DebugString() const;
};

}  // namespace tsl

#endif  // XLA_TSL_FRAMEWORK_ALLOCATOR_H_
