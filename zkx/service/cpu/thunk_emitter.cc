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

#include "zkx/service/cpu/thunk_emitter.h"

#include "absl/strings/str_format.h"

#include "xla/tsl/platform/casts.h"
#include "zkx/backends/cpu/codegen/elemental/elemental_kernel_emitter.h"
#include "zkx/backends/cpu/runtime/kernel_thunk.h"
#include "zkx/codegen/llvm_ir_kernel_source.h"
#include "zkx/cpu_function_runtime.h"

namespace zkx::cpu {
namespace {

Thunk::Info ThunkInfo(const HloInstruction* instruction) {
  const HloModule* module = instruction->GetModule();
  return Thunk::Info{std::string(instruction->name()),
                     std::string(module->name()), module->unique_id()};
}

}  // namespace

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHloInstruction(
    const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kParameter:
      return ThunkSequence::Empty();
    case HloOpcode::kAdd:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
      return EmitElementalKernelThunk(instr);

    default:
      return absl::InternalError(
          absl::StrFormat("Unsupported instruction opcode: %s",
                          HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitHloComputation(
    const HloComputation* computation) {
  ThunkSequence thunks;

  const HloSchedule& schedule = computation->parent()->schedule();
  if (!schedule.is_computation_scheduled(computation))
    return absl::InternalError(absl::StrFormat(
        "Sequence not found for computation: %s", computation->name()));

  const HloInstructionSequence& sequence = schedule.sequence(computation);
  for (HloInstruction* instr : sequence.instructions()) {
    TF_ASSIGN_OR_RETURN(ThunkSequence instr_thunks, EmitHloInstruction(instr));
    thunks.Append(std::move(instr_thunks));
  }

  return thunks;
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitEntryComputation(
    const HloModule& module) {
  if (!module.has_schedule()) {
    return absl::InternalError("HLO module must be scheduled to emit thunks");
  }
  // TODO(chokobole): Uncomment this. Dependency: Profiler
  // tsl::profiler::TraceMe trace("ThunkEmitter::EmitEntryComputation");
  return EmitHloComputation(module.entry_computation());
}

absl::StatusOr<ThunkSequence> ThunkEmitter::EmitElementalKernelThunk(
    const HloInstruction* instruction) {
  ElementalKernelEmitter emitter(mlir_context_, instruction,
                                 buffer_assignment_);
  TF_ASSIGN_OR_RETURN(KernelDefinition kernel_definition,
                      emitter.EmitKernelDefinition());

  auto [kernel_spec, kernel_source] = std::move(kernel_definition).release();
  auto llvm_ir_kernel_source = absl::WrapUnique<LlvmIrKernelSource>(
      tsl::down_cast<LlvmIrKernelSource*>(kernel_source.release()));

  kernels_.push_back({kernel_spec.name(),
                      std::move(*llvm_ir_kernel_source).thread_safe_module()});

  return MakeKernelThunkSequence(
      instruction, std::move(kernel_spec),
      /*min_alignment=*/cpu_function_runtime::MinAlign());
}

// static
absl::StatusOr<ThunkSequence> ThunkEmitter::MakeKernelThunkSequence(
    const HloInstruction* instruction, const KernelSpec& kernel_spec,
    std::optional<uint64_t> min_alignment) {
  return ThunkSequence::Of<KernelThunk>(ThunkInfo(instruction), kernel_spec,
                                        min_alignment);
}

}  // namespace zkx::cpu
