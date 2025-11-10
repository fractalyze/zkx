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

#include "zkx/backends/cpu/runtime/kernel.h"

#include <algorithm>
#include <limits>

#include "absl/debugging/leak_check.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace zkx::cpu {
namespace {

using LaunchEvent = Kernel::LaunchEvent;

// Non-reference-counted async value ref for host kernels executed inline.
tsl::AsyncValueRef<LaunchEvent> OkLaunchEvent() {
  static tsl::AsyncValueOwningRef<LaunchEvent>* event = [] {
    auto* storage =
        absl::IgnoreLeak(new tsl::internal::AsyncValueStorage<LaunchEvent>());
    return absl::IgnoreLeak(new tsl::AsyncValueOwningRef<LaunchEvent>(
        tsl::MakeAvailableAsyncValueRef<LaunchEvent>(*storage)));
  }();
  return event->AsRef();
}

absl::InlinedVector<ZKX_CPU_KernelArg, 8> ConvertBuffersToKernelArgs(
    absl::Span<const Kernel::DeviceMemoryBase> buffers) {
  absl::InlinedVector<ZKX_CPU_KernelArg, 8> args(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    args[i].data = const_cast<void*>(buffers[i].opaque());
    args[i].size = buffers[i].size();
  }
  return args;
}

}  // namespace

template <bool kThreadDimXOnly>
class Kernel::ParallelTask {
 public:
  ParallelTask(ZKX_CPU_Kernel* kernel, Kernel::ThreadDim thread_dims,
               absl::Span<const ZKX_CPU_KernelArg> args)
      : kernel_(kernel),
        thread_dims_({thread_dims.x, thread_dims.y, thread_dims.z}),
        args_(args.begin(), args.end()),
        num_tasks_(thread_dims_.x * thread_dims_.y * thread_dims_.z),
        stride_z_(thread_dims_.y * thread_dims_.x),
        stride_y_(thread_dims_.x) {}

  // Invokes a host kernel for a given task index.
  absl::Status operator()(size_t task_index) const;

 private:
  // Converts linear task index in [0, num_tasks) to (x, y, z) coordinate. We
  // assume that `x` is the fastest iterating dimension.
  ZKX_CPU_KernelThread Delinearize(uint64_t task_index) const;

  ZKX_CPU_Kernel* kernel_;
  ZKX_CPU_KernelThreadDim thread_dims_;
  absl::InlinedVector<ZKX_CPU_KernelArg, 8> args_;

  size_t num_tasks_;

  // Strides for delinearizing task index to (x, y, z) coordinate.
  uint64_t stride_z_;
  uint64_t stride_y_;
};

template <bool kThreadDimXOnly>
absl::Status Kernel::ParallelTask<kThreadDimXOnly>::operator()(
    size_t task_index) const {
  DCHECK_LT(task_index, num_tasks_) << "Task index out of range";  // Crash OK

  ZKX_CPU_KernelThread kernel_thread = Delinearize(task_index);
  ZKX_CPU_KernelCallFrame call_frame = {&thread_dims_, &kernel_thread,
                                        args_.size(), args_.data()};

  ZKX_CPU_KernelError* error = (*kernel_)(&call_frame);

  if (ABSL_PREDICT_TRUE(error == nullptr)) {
    return absl::OkStatus();
  } else {
    return absl::InternalError(
        absl::StrFormat("Failed to call host kernel: x=%d, y=%d, z=%d",
                        kernel_thread.x, kernel_thread.y, kernel_thread.z));
  }
}

template <bool kThreadDimXOnly>
ZKX_CPU_KernelThread Kernel::ParallelTask<kThreadDimXOnly>::Delinearize(
    uint64_t task_index) const {
  // In the most common case we parallelize only over the `x` dimension.
  if constexpr (kThreadDimXOnly) {
    return ZKX_CPU_KernelThread{task_index, 1, 1};
  }

  // Convert linear task index to (x, y, z) coordinate.
  uint64_t z = task_index / stride_z_;
  task_index = task_index % stride_z_;
  uint64_t y = task_index / stride_y_;
  task_index = task_index % stride_y_;
  uint64_t x = task_index;

  return ZKX_CPU_KernelThread{x, y, z};
}

absl::Status Kernel::Launch(const ThreadDim& thread_dims,
                            absl::Span<const DeviceMemoryBase> buffers) const {
  return Launch(thread_dims, ConvertBuffersToKernelArgs(buffers));
}

absl::Status Kernel::Launch(const ThreadDim& thread_dims,
                            absl::Span<const ZKX_CPU_KernelArg> args) const {
  ZKX_CPU_KernelThreadDim kernel_thread_dims = {
      thread_dims.x,
      thread_dims.y,
      thread_dims.z,
  };

  for (uint64_t z = 0; z < thread_dims.z; ++z) {
    for (uint64_t y = 0; y < thread_dims.y; ++y) {
      for (uint64_t x = 0; x < thread_dims.x; ++x) {
        ZKX_CPU_KernelThread kernel_thread = {x, y, z};

        ZKX_CPU_KernelCallFrame call_frame = {
            &kernel_thread_dims, &kernel_thread, args.size(), args.data()};

        ZKX_CPU_KernelError* error = (*kernel_)(&call_frame);

        if (ABSL_PREDICT_FALSE(error != nullptr)) {
          return absl::InternalError("Failed to call host kernel");
        }
      }
    }
  }

  return absl::OkStatus();
}

tsl::AsyncValueRef<LaunchEvent> Kernel::Launch(
    const ThreadDim& thread_dims, absl::Span<const DeviceMemoryBase> buffers,
    const Eigen::ThreadPoolDevice* device) const {
  return Launch(thread_dims, ConvertBuffersToKernelArgs(buffers), device);
}

tsl::AsyncValueRef<LaunchEvent> Kernel::Launch(
    const ThreadDim& thread_dims, absl::Span<const ZKX_CPU_KernelArg> args,
    const Eigen::ThreadPoolDevice* device) const {
  size_t num_tasks = thread_dims.x * thread_dims.y * thread_dims.z;
  CHECK_GT(num_tasks, 0) << "Number of tasks must be positive";  // Crash Ok

  // Short-circuit launch with a single task and run it in the caller thread.
  if (ABSL_PREDICT_TRUE(num_tasks == 1)) {
    absl::Status launched = Launch(thread_dims, args);
    return ABSL_PREDICT_TRUE(launched.ok())
               ? OkLaunchEvent()
               : tsl::MakeErrorAsyncValueRef(std::move(launched));
  }

  // Do not create more workers than the number of threads in the thread pool.
  size_t num_workers =
      std::min<size_t>(std::min<size_t>(num_tasks, device->numThreadsInPool()),
                       std::numeric_limits<uint16_t>::max());

  if (ABSL_PREDICT_TRUE(thread_dims.y == 1 && thread_dims.z == 1)) {
    return Worker::Parallelize(device, num_workers, num_tasks,
                               ParallelTask<true>(kernel_, thread_dims, args));
  } else {
    return Worker::Parallelize(device, num_workers, num_tasks,
                               ParallelTask<false>(kernel_, thread_dims, args));
  }
}

}  // namespace zkx::cpu
