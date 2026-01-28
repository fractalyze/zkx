/* Copyright 2018 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_GPU_GPU_FUSIBLE_H_
#define ZKX_SERVICE_GPU_GPU_FUSIBLE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/service/gpu/hlo_fusion_analysis.h"
#include "zkx/service/gpu/launch_dimensions.h"
#include "zkx/service/instruction_fusion.h"
#include "zkx/stream_executor/device_description.h"

namespace zkx::gpu {

// Fusion passes frequently do checks across all pairs of "interesting" nodes.
// Computing e.g. FusionFitsInBudget(a, b) requires computing expensive
// properties of `a` and `b` individually.  This cache lets us avoid recomputing
// those properties n² times.
//
// Invariant: After modifying or removing a fusion node, call Invalidate(node).
class FusionInfoCache {
 public:
  explicit FusionInfoCache(const se::DeviceDescription& device_info)
      : device_info_(device_info) {}
  // Must be called after modifying or removing a fusion node (or other node
  // that's part of this cache).
  void Invalidate(const HloInstruction* instr) {
    shared_memory_usage_.erase(instr);
    num_unnested_reductions_.erase(instr);
  }

  // Returns expected shared memory usage of a given instruction in bytes.
  int64_t GetSharedMemoryUsage(const HloInstruction& instr);

  // Returns the number of unnested reductions in the instruction output.
  int64_t GetNumUnnestedReductions(const HloInstruction& instr);

 private:
  const se::DeviceDescription& device_info_;

  absl::Mutex mutex_;

  absl::flat_hash_map<const HloInstruction*, int64_t> shared_memory_usage_;
  absl::flat_hash_map<const HloInstruction*, int64_t> num_unnested_reductions_;
};

// Returns the max loop unroll factor.
inline constexpr int64_t MaxUnrollFactor() { return 4; }

inline constexpr int64_t MaxOperandsAndOutputsPerFusion() { return 96; }

// Whether the op transposes the physical data layout. Fusing such ops may lead
// to uncoalesced data access and may thus not be beneficial.
bool IsPhysicallyTransposing(const HloInstruction& instr);

// Whether the op transposes the minor-most dimension. In the case of fusions,
// whether the fusion contains some op that does this.
// If the minor-most dimension is transposed, this results in uncoalesced memory
// accesses in untiled code generators.
bool TransposesMinorDimension(const HloInstruction* instr);

// Whether `instr` is an input fusion rooted at a reduction-to-vector op or a
// multi-output input fusion with at least one reduction-to-vector op root.
bool IsReduceInputFusion(const HloInstruction& instr,
                         const se::DeviceDescription& device_info);

// Whether `instr` is fusible as root of a reduce input fusions, i.e. `instr`
// is either an unfused reduction-to-vector op or a reduce input fusion.
bool IsInputFusibleReduction(const HloInstruction& instr,
                             const se::DeviceDescription& device_info);

// Returns instructions which are roots of the fusion, following the operands of
// GTE instructions in the root tuple that extract from a tuple.
std::vector<const HloInstruction*> GetFusionRoots(
    const HloComputation& computation);

// Returns a list of computations within `module` that are candidates for fusion
// using priority fusion.
std::vector<HloComputation*> GetFusibleComputations(
    const HloModule& module,
    const absl::flat_hash_set<std::string_view>& execution_threads);

// Check if fusing producer and consumer will exceed the resource budget.
FusionDecision FusionFitsInBudget(const HloInstruction& instr1,
                                  const HloInstruction& instr2,
                                  const se::DeviceDescription& device_info,
                                  bool is_consumer_producer_fusion = false,
                                  FusionInfoCache* cache = nullptr);

// Check if fusing the scatter with the producer is supported.
FusionDecision CanEmitInputFusedScatter(const HloInstruction& producer,
                                        const HloInstruction& consumer);

LaunchDimensionsConfig ComputeLoopFusionConfig(
    const HloFusionAnalysis& analysis);

LaunchDimensionsConfig ComputeLoopFusionConfig(
    const HloFusionAnalysis& analysis, const Shape& shape);

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_GPU_FUSIBLE_H_
