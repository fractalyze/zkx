/* Copyright 2024 The OpenXLA Authors.
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

#include "zkx/service/gpu/transforms/scheduling_instruction_annotator.h"

#include "absl/log/check.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"

namespace zkx::gpu {
namespace {

// Populates `OpMetadata`'s `scheduling_name` field for all of the instructions
// belonging to `computation`.
absl::StatusOr<bool> AnnotateSchedulingInstructionNames(
    HloComputation& computation) {
  bool changed = false;
  for (HloInstruction* inst : computation.instructions()) {
    if (!inst->metadata().scheduling_name().empty()) {
      continue;
    }
    // We skip constants as we might have to sanitize them in order to satisfy
    // LLVM backend. I.e. we allow `GpuSanitizeConstantNames` pass to run post
    // scheduling.
    if (HloPredicateIsOp<HloOpcode::kConstant>(inst)) {
      continue;
    }
    inst->set_metadata_scheduling_name(inst->name());
    changed = true;
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> SchedulingInstructionAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  CHECK(module->has_schedule())
      << "The pass is supposed to run in the beginning of post-scheduling!";
  bool changed = false;

  // We visit computations in the order of callees to callers, as information is
  // propagated from callees to callers.
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result,
                        AnnotateSchedulingInstructionNames(*computation));
    changed |= result;
  }

  return changed;
}

}  // namespace zkx::gpu
