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

#ifndef ZKX_BACKENDS_CPU_RUNTIME_KERNEL_H_
#define ZKX_BACKENDS_CPU_RUNTIME_KERNEL_H_

#include <memory>
#include <type_traits>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "zkx/backends/cpu/runtime/kernel_c_api.h"
#include "zkx/backends/cpu/runtime/work_queue.h"
#include "zkx/stream_executor/device_memory.h"
#include "zkx/stream_executor/launch_dim.h"

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace zkx::cpu {

class Kernel {
 public:
  // A struct to report completion of the kernel execution.
  using LaunchEvent = tsl::Chain;

  using ThreadDim = se::ThreadDim;
  using DeviceMemoryBase = se::DeviceMemoryBase;

  // Virtual base class that owns the function behind the host kernel. It can be
  // a function in a jit-compiled LLVM module or simply a pointer to the
  // in-process function written in C++. Kernel is responsible for launching
  // the kernel function owned by the KernelFunction with given user-provided
  // arguments potentially on a thread pool.
  class KernelFunction {
   public:
    virtual ~KernelFunction() = default;
    virtual ZKX_CPU_Kernel* kernel() const = 0;
  };

  // A wrapper around function pointer that implements ZKX_CPU_Kernel API.
  class KernelFunctionPtr final : public KernelFunction {
   public:
    explicit KernelFunctionPtr(ZKX_CPU_Kernel* ptr) : ptr_(ptr) {}
    ZKX_CPU_Kernel* kernel() const final { return ptr_; }

   private:
    ZKX_CPU_Kernel* ptr_;  // not owned
  };

  // TODO(tsilytskyi): make this implementation detail private
  Kernel(unsigned arity, ZKX_CPU_Kernel* kernel)
      : function_(std::make_unique<KernelFunctionPtr>(kernel)),
        kernel_(function_->kernel()),
        arity_(arity) {}

  // Calls the kernel once in the caller thread for a thread dim (0,0,0).
  // This is a fast path for small host kernels that have just one thread.
  absl::Status CallOnce(absl::Span<const ZKX_CPU_KernelArg> args) const;

  // Launches the kernel on the current thread by iterating over all threads in
  // `thread_dims` and calling the kernel function.
  absl::Status Launch(const ThreadDim& thread_dims,
                      absl::Span<const DeviceMemoryBase> buffers) const;
  absl::Status Launch(const ThreadDim& thread_dims,
                      absl::Span<const ZKX_CPU_KernelArg> args) const;

  // Launches the kernel by iterating over all threads in `thread_dims` and
  // using `device` to parallelize the execution.
  //
  // The returned async value becomes available after all tasks are completed.
  // Async value returned in constructed state and the caller can access it to
  // get the number of tasks that are expected to be completed.
  tsl::AsyncValueRef<LaunchEvent> Launch(
      const ThreadDim& thread_dims, absl::Span<const DeviceMemoryBase> buffers,
      const Eigen::ThreadPoolDevice* device) const;
  tsl::AsyncValueRef<LaunchEvent> Launch(
      const ThreadDim& thread_dims, absl::Span<const ZKX_CPU_KernelArg> args,
      const Eigen::ThreadPoolDevice* device) const;

  // For host platform, we assume that a core is a thread, and we can run at
  // most one instance of a kernel on a given thread.
  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(ThreadDim, size_t) const {
    return 1;
  }

  void SetArity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const { return arity_; }

  template <typename T,
            std::enable_if_t<std::is_base_of_v<KernelFunction, T>>* = nullptr>
  void SetKernelFunction(std::unique_ptr<T> function) {
    function_ = std::move(function);
    kernel_ = function_->kernel();
  }

 private:
  // A kernel parallel task that is used to parallelize host kernel execution.
  template <bool kThreadDimXOnly>
  class ParallelTask;

  std::unique_ptr<KernelFunction> function_;
  ZKX_CPU_Kernel* kernel_;  // pointer to the kernel owned by `function_`

  unsigned arity_;
};

inline ABSL_ATTRIBUTE_ALWAYS_INLINE absl::Status Kernel::CallOnce(
    absl::Span<const ZKX_CPU_KernelArg> args) const {
  constexpr ZKX_CPU_KernelThreadDim kernel_thread_dims = {1, 1, 1};
  constexpr ZKX_CPU_KernelThread kernel_thread = {1, 1, 1};

  ZKX_CPU_KernelCallFrame call_frame = {&kernel_thread_dims, &kernel_thread,
                                        args.size(), args.data()};

  ZKX_CPU_KernelError* error = (*kernel_)(&call_frame);

  if (ABSL_PREDICT_FALSE(error != nullptr)) {
    return absl::InternalError("Failed to call host kernel");
  }

  return absl::OkStatus();
}

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_RUNTIME_KERNEL_H_
