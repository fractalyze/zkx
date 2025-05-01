#ifndef ZKX_SERVICE_CPU_CPU_EXECUTABLE_H_
#define ZKX_SERVICE_CPU_CPU_EXECUTABLE_H_

#include <memory>
#include <optional>

#include "zkx/backends/cpu/runtime/function_library.h"
#include "zkx/backends/cpu/runtime/thunk.h"
#include "zkx/backends/cpu/runtime/thunk_executor.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/service/buffer_assignment.h"

namespace zkx::cpu {

class CpuExecutable {
 public:
  // Creates a CpuExecutable from a thunk sequence.
  static absl::StatusOr<std::unique_ptr<CpuExecutable>> Create(
      std::unique_ptr<FunctionLibrary> function_library,
      std::unique_ptr<const BufferAssignment> assignment,
      std::unique_ptr<HloModule> hlo_module, ThunkSequence thunks);

  bool has_thunks() const { return thunks_.has_value(); }
  ThunkExecutor& thunks() { return *thunks_; }

  const BufferAssignment& buffer_assignment() const { return *assignment_; }

  absl::Span<const BufferAllocation> GetAllocations() const {
    return assignment_->Allocations();
  }

  FunctionLibrary* function_library() const { return function_library_.get(); }

 private:
  CpuExecutable(std::unique_ptr<HloModule> hlo_module,
                std::unique_ptr<const BufferAssignment> assignment);
  CpuExecutable(const CpuExecutable&) = delete;
  CpuExecutable& operator=(const CpuExecutable&) = delete;

  // A thunk executor created from the compiled thunk sequence.
  std::unique_ptr<HloModule> hlo_module_;

  // The FunctionLibrary containing compiled modules.
  std::unique_ptr<FunctionLibrary> function_library_;

  // Buffer assignment for the buffers we need to allocate.
  const std::unique_ptr<const BufferAssignment> assignment_;

  std::optional<ThunkExecutor> thunks_;
};

}  // namespace zkx::cpu

#endif  // ZKX_SERVICE_CPU_CPU_EXECUTABLE_H_
