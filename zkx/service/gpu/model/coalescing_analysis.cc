/* Copyright 2023 The OpenXLA Authors.
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

#include "zkx/service/gpu/model/coalescing_analysis.h"

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/utils/hlo_traversal.h"
#include "zkx/service/gpu/gpu_fusible.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {

// Returns true if all input reads are coalesced. If consumer is not nullptr,
// producer and consumer are considered as one fusion, otherwise it's only the
// producer.
bool IsReadCoalescedHeuristic(HloFusionAnalysis::EmitterFusionKind fusion_kind,
                              const se::DeviceDescription& device_info,
                              const HloInstruction* producer,
                              const HloInstruction* consumer) {
  // Transposing minor dimension breaks coalescing.
  if (fusion_kind != HloFusionAnalysis::EmitterFusionKind::kTranspose) {
    auto is_broadcast = [&](const HloInstruction* instr) {
      while (true) {
        if (instr->opcode() == HloOpcode::kBroadcast ||
            instr->opcode() == HloOpcode::kIota) {
          return true;
        }
        if (instr->operand_count() != 1) return false;
        if (instr->opcode() != HloOpcode::kBitcast && !instr->IsElementwise()) {
          return false;
        }
        instr = instr->operand(0);
      }
    };
    auto is_bad_transpose = [&](const HloInstruction* instr) {
      if (instr->opcode() == HloOpcode::kFusion) {
        for (auto* fused_instr : instr->fused_instructions()) {
          // Hack: we allow transposes of broadcasts or iotas.
          if (TransposesMinorDimension(fused_instr) &&
              !is_broadcast(fused_instr->operand(0))) {
            return true;
          }
        }
        return false;
      }
      // Hack: we allow transposes of broadcasts or iotas.
      return TransposesMinorDimension(instr) &&
             !is_broadcast(instr->operand(0));
    };
    if (is_bad_transpose(producer)) return false;
    if (consumer && is_bad_transpose(consumer)) return false;
  }
  // Fusing two row reductions breaks coalescing.
  if (fusion_kind == HloFusionAnalysis::EmitterFusionKind::kReduction &&
      IsInputFusibleReduction(*producer, device_info) && consumer &&
      IsInputFusibleReduction(*consumer, device_info)) {
    return false;
  }
  return true;
}

CoalescingAnalysis::CoalescingAnalysis(
    const HloInstruction* instr,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis) {
  // Use heuristic-only approach. Mark all operands as coalesced if the
  // heuristic says they are.
  is_coalesced_computed_by_heuristic_ =
      IsReadCoalescedHeuristic(fusion_analysis.GetEmitterFusionKind(),
                               fusion_analysis.device_info(), instr);
}

CoalescingAnalysis::CoalescingAnalysis(
    const HloInstruction* producer, const HloInstruction* consumer,
    absl::Span<const HloInstruction* const> operands,
    const HloFusionAnalysis& fusion_analysis) {
  // Use heuristic-only approach. Mark all operands as coalesced if the
  // heuristic says they are.
  is_coalesced_computed_by_heuristic_ = IsReadCoalescedHeuristic(
      fusion_analysis.GetEmitterFusionKind(), fusion_analysis.device_info(),
      producer, consumer);
}

bool CoalescingAnalysis::IsReadCoalesced(const HloInstruction* operand) const {
  auto it = coalescing_per_operand_.find(operand);
  if (it == coalescing_per_operand_.end()) {
    return is_coalesced_computed_by_heuristic_;
  }
  return it->second;
}

}  // namespace zkx::gpu
