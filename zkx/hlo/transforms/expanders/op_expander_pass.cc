/* Copyright 2018 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#include "zkx/hlo/transforms/expanders/op_expander_pass.h"

#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"

#include "xla/tsl/platform/statusor.h"

namespace zkx {

absl::StatusOr<bool> OpExpanderPass::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  std::vector<HloInstruction*> matching_instructions;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    absl::c_copy_if(computation->MakeInstructionPostOrder(),
                    std::back_inserter(matching_instructions),
                    [&](HloInstruction* inst) {
                      return InstructionMatchesPattern(inst) &&
                             (!extra_filter_ || extra_filter_(inst));
                    });
  }

  for (HloInstruction* inst : matching_instructions) {
    TF_ASSIGN_OR_RETURN(HloInstruction * expanded_root,
                        ExpandInstruction(inst));
    if (expanded_root == nullptr) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(bool changed,
                        inst->parent()->ReplaceInstruction(
                            inst, expanded_root, preserve_sharding_,
                            relay_control_dependency_));
    // NOTE(chokobole): `ReplaceInstruction()` might return `false` if `inst`
    // has already been removed or replaced by a previous transformation in this
    // loop. We omit the `DCHECK()` to avoid crashes on instructions that became
    // dead during the pass.
    // See https://github.com/fractalyze/zkx/pull/172#discussion_r2675302613
    std::ignore = changed;
  }

  return !matching_instructions.empty();
}
}  // namespace zkx
