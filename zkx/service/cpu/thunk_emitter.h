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

#ifndef ZKX_SERVICE_CPU_THUNK_EMITTER_H_
#define ZKX_SERVICE_CPU_THUNK_EMITTER_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "mlir/IR/MLIRContext.h"

#include "zkx/backends/cpu/codegen/jit_compiler.h"
#include "zkx/backends/cpu/runtime/thunk.h"
#include "zkx/codegen/kernel_spec.h"
#include "zkx/service/buffer_assignment.h"

namespace zkx::cpu {

class ThunkEmitter {
 public:
  struct EmittedKernel {
    std::string kernel_name;
    llvm::orc::ThreadSafeModule module;
  };

  ThunkEmitter(const BufferAssignment* buffer_assignment,
               mlir::MLIRContext* mlir_context)
      : buffer_assignment_(buffer_assignment), mlir_context_(mlir_context) {}

  absl::StatusOr<ThunkSequence> EmitEntryComputation(const HloModule& module);

  std::vector<EmittedKernel>& kernels() { return kernels_; }

 private:
  struct HostKernelAllocationSlices {
    std::vector<BufferAllocation::Slice> arguments;
    std::vector<BufferAllocation::Slice> results;
  };

  absl::StatusOr<ThunkSequence> EmitHloComputation(
      const HloComputation* computation);
  absl::StatusOr<ThunkSequence> EmitHloInstruction(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitElementalKernelThunk(
      const HloInstruction* instruction);

  static absl::StatusOr<ThunkSequence> MakeKernelThunkSequence(
      const HloInstruction* instruction, const KernelSpec& kernel_spec,
      std::optional<uint64_t> min_alignment = std::nullopt);

  const BufferAssignment* const buffer_assignment_;
  mlir::MLIRContext* const mlir_context_;
  std::vector<EmittedKernel> kernels_;
};

}  // namespace zkx::cpu

#endif  // ZKX_SERVICE_CPU_THUNK_EMITTER_H_
