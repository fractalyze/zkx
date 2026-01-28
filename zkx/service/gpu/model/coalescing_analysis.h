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

#ifndef ZKX_SERVICE_GPU_MODEL_COALESCING_ANALYSIS_H_
#define ZKX_SERVICE_GPU_MODEL_COALESCING_ANALYSIS_H_

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {

// Computes read coalescing for operands of an instruction or a
// producer-consumer fusion.
//
// This is a simplified implementation that uses only heuristics, without
// the MLIR indexing analysis paths present in XLA.
class CoalescingAnalysis {
 public:
  // Computes read coalescing for operands of `instr`.
  CoalescingAnalysis(const HloInstruction* instr,
                     absl::Span<const HloInstruction* const> operands,
                     const HloFusionAnalysis& fusion_analysis);

  // Computes read coalescing for operands of fused `producer` and `consumer`.
  CoalescingAnalysis(const HloInstruction* producer,
                     const HloInstruction* consumer,
                     absl::Span<const HloInstruction* const> operands,
                     const HloFusionAnalysis& fusion_analysis);

  // Returns true if the operand is read coalesced.
  bool IsReadCoalesced(const HloInstruction* operand) const;

 private:
  absl::flat_hash_map<const HloInstruction*, bool> coalescing_per_operand_;
  bool is_coalesced_computed_by_heuristic_ = false;
};

// Returns true if all input reads are coalesced. If consumer is not nullptr,
// producer and consumer are considered as one fusion, otherwise it's only the
// producer.
bool IsReadCoalescedHeuristic(HloFusionAnalysis::EmitterFusionKind fusion_kind,
                              const se::DeviceDescription& device_info,
                              const HloInstruction* producer,
                              const HloInstruction* consumer = nullptr);

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_MODEL_COALESCING_ANALYSIS_H_
