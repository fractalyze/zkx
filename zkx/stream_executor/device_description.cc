/* Copyright 2015 The OpenXLA Authors.

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

#include "zkx/stream_executor/device_description.h"

#include "absl/log/check.h"

#include "xla/tsl/lib/math/math_util.h"
#include "zkx/base/logging.h"

namespace stream_executor {

const GpuComputeCapability& DeviceDescription::gpu_compute_capability() const {
  return gpu_compute_capability_;
}

CudaComputeCapability DeviceDescription::cuda_compute_capability() const {
  if (auto* ptr =
          std::get_if<CudaComputeCapability>(&gpu_compute_capability_)) {
    return *ptr;
  }
  // Fallback for backwards compatibility.
  return CudaComputeCapability{-1, -1};
}

RocmComputeCapability DeviceDescription::rocm_compute_capability() const {
  if (auto* ptr =
          std::get_if<RocmComputeCapability>(&gpu_compute_capability_)) {
    return *ptr;
  }
  return RocmComputeCapability{};
}

bool ThreadDimOk(const DeviceDescription& device_description,
                 const ThreadDim& thread_dim) {
  const int64_t total_threads = thread_dim.x * thread_dim.y * thread_dim.z;
  const int64_t threads_per_block_limit =
      device_description.threads_per_block_limit();
  if (total_threads > threads_per_block_limit) {
    VLOG(2) << "exceeded total-thread-per-block limit: " << total_threads
            << " vs limit " << threads_per_block_limit;
    return false;
  }

  const auto& limit = device_description.thread_dim_limit();
  bool ok = thread_dim.x <= limit.x && thread_dim.y <= limit.y &&
            thread_dim.z <= limit.z;
  if (!ok) {
    VLOG(2) << "thread dim " << thread_dim.ToString()
            << " exceeds limit constraints of " << limit.ToString();
  }
  return ok;
}

void CalculateDimensionality(const DeviceDescription& device_description,
                             int64_t element_count, int64_t* threads_per_block,
                             int64_t* block_count) {
  *threads_per_block = device_description.threads_per_block_limit();
  *block_count = tsl::MathUtil::CeilOfRatio(element_count, *threads_per_block);
  if (*block_count == 1) {
    CHECK_LE(element_count, *threads_per_block);
    *threads_per_block = element_count;
  }
}

}  // namespace stream_executor
