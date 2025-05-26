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

#include "xla/tsl/framework/allocator.h"

#include "absl/strings/str_format.h"

namespace tsl {

std::string AllocatorStats::DebugString() const {
  return absl::StrFormat(
      "Limit:            %20lld\n"
      "InUse:            %20lld\n"
      "MaxInUse:         %20lld\n"
      "NumAllocs:        %20lld\n"
      "MaxAllocSize:     %20lld\n"
      "Reserved:         %20lld\n"
      "PeakReserved:     %20lld\n"
      "LargestFreeBlock: %20lld\n",
      static_cast<long long>(this->bytes_limit ? *this->bytes_limit : 0),
      static_cast<long long>(this->bytes_in_use),
      static_cast<long long>(this->peak_bytes_in_use),
      static_cast<long long>(this->num_allocs),
      static_cast<long long>(this->largest_alloc_size),
      static_cast<long long>(this->bytes_reserved),
      static_cast<long long>(this->peak_bytes_reserved),
      static_cast<long long>(this->largest_free_block_bytes));
}

}  // namespace tsl
