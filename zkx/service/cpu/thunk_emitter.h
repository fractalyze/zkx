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

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
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
               mlir::MLIRContext* mlir_context);

  absl::StatusOr<ThunkSequence> EmitEntryComputation(const HloModule& module);

  std::vector<EmittedKernel>& kernels() { return kernels_; }

 private:
  struct HostKernelAllocationSlices {
    std::vector<BufferAllocation::Slice> arguments;
    std::vector<BufferAllocation::Slice> results;
  };

  // Returns the buffer allocation slice assigned to the given instruction at
  // the given shape index. Instruction must have a unique slice assigned to it!
  absl::StatusOr<BufferAllocation::Slice> GetAllocationSlice(
      const HloInstruction* instruction, const ShapeIndex& index = {});

  // Returns a token resource corresponding to the given instruction result.
  absl::StatusOr<std::shared_ptr<Resource>> GetTokenResource(
      const HloInstruction* instruction, const ShapeIndex& index = {});

  absl::StatusOr<ThunkSequence> EmitHloComputation(
      const HloComputation* computation);
  absl::StatusOr<ThunkSequence> EmitHloInstruction(const HloInstruction* instr);

  absl::StatusOr<ThunkSequence> EmitAllGatherThunk(
      const HloInstruction* instruction);
  absl::StatusOr<ThunkSequence> EmitAllReduceThunk(
      const HloInstruction* instruction);
  absl::StatusOr<ThunkSequence> EmitAllToAllThunk(
      const HloInstruction* instruction);
  absl::StatusOr<ThunkSequence> EmitCollectivePermuteThunk(
      const HloInstruction* instruction);
  absl::StatusOr<ThunkSequence> EmitReduceScatterThunk(
      const HloInstruction* instruction);
  absl::StatusOr<ThunkSequence> EmitInfeedThunk(
      const HloInstruction* instruction);
  absl::StatusOr<ThunkSequence> EmitOutfeedThunk(
      const HloInstruction* instruction);
  absl::StatusOr<ThunkSequence> EmitCopyThunk(
      const HloInstruction* instruction);
  absl::StatusOr<ThunkSequence> EmitKernelThunk(
      const HloInstruction* instruction);

  static absl::StatusOr<ThunkSequence> MakeKernelThunkSequence(
      const HloInstruction* instruction, const KernelSpec& kernel_spec,
      std::optional<uint64_t> min_alignment = std::nullopt);

  const BufferAssignment* const buffer_assignment_;
  mlir::MLIRContext* const mlir_context_;

  // Token resources that correspond to the token buffer allocation slices. We
  // rely on buffer assignment to assign unique "identity" to each token, and
  // create a separate resource for each unique allocation slice.
  absl::flat_hash_map<BufferAllocation::Slice, std::shared_ptr<Resource>>
      token_resources_;

  std::vector<EmittedKernel> kernels_;

  // A global resource that is used to order all collective operations.
  std::shared_ptr<Resource> communicator_resource_;
};

}  // namespace zkx::cpu

#endif  // ZKX_SERVICE_CPU_THUNK_EMITTER_H_
