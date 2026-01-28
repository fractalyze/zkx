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

#include "zkx/service/gpu/gpu_fusible.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/utils/hlo_traversal.h"
#include "zkx/permutation_util.h"
#include "zkx/service/gpu/ir_emission_utils.h"
#include "zkx/service/gpu/reduction_utils.h"
#include "zkx/service/instruction_fusion.h"
#include "zkx/shape_util.h"
#include "zkx/util.h"

namespace zkx::gpu {
namespace {

const Shape& GetElementShape(const HloFusionAnalysis& analysis) {
  const Shape* shape = &analysis.fusion_root(0).shape();
  while (shape->IsTuple()) {
    shape = &shape->tuple_shapes(0);
  }
  return *shape;
}

// Computes the maximum valid unroll factor for a given instruction.
int ComputeMaxUnrollFactor(int64_t num_elements) {
  for (int i = MaxUnrollFactor(); i > 1; i /= 2) {
    if (num_elements % i == 0) {
      return i;
    }
  }
  return 1;
}

// Whether the fusion will likely behave poorly with vectorization due to the
// instructions it contains.
bool MayPreventVectorization(const HloFusionAdaptor& fusion) {
  // An empirically chosen constant: unrolling concat with a large amount of
  // arguments causes excessive register spilling.
  static constexpr int kMaxConcatArgumentsForUnrolling = 10;
  return HloAnyOf(fusion, [&](auto node) {
    switch (node.opcode()) {
      case HloOpcode::kDot:
      case HloOpcode::kSort:
        return true;
      case HloOpcode::kConcatenate:
        return node.instruction().operand_count() >
               kMaxConcatArgumentsForUnrolling;
      case HloOpcode::kReduce:
        return node.instruction().shape().tuple_shapes_size() > 1;
      default:
        return false;
    }
  });
}

std::pair<int64_t, int64_t> MostMinorNonTrivialDimension(const Shape& shape) {
  int64_t position_of_first_non_trivial_dim = 0;
  for (int64_t dim : shape.layout().minor_to_major()) {
    if (shape.dimensions()[dim] > 1) {
      return {dim, position_of_first_non_trivial_dim};
    }
    ++position_of_first_non_trivial_dim;
  }
  return {-1, position_of_first_non_trivial_dim};
}

// Returns an estimate of the shared memory usage for a given instruction in
// bytes.
int64_t SharedMemoryUsageNoCache(const HloInstruction& instr,
                                 const se::DeviceDescription& device_info) {
  if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->instructions()) {
      sum += SharedMemoryUsageNoCache(*hlo, device_info);
    }
    return sum;
  } else if (instr.opcode() == HloOpcode::kReduce &&
             IsReductionFromOrToContiguousDimensions(instr, device_info)) {
    ReductionDimensions reduction_info =
        GetReductionKindAndContiguousComponents(instr);
    int64_t primitive_size = ShapeUtil::ByteSizeOfPrimitiveType(
        instr.operand(0)->shape().element_type());
    int num_variadic =
        instr.shape().IsTuple() ? instr.shape().tuple_shapes_size() : 1;
    if (reduction_info.is_row_reduction) {
      // __shared__[32] is used for row reduction.
      return 32 * primitive_size * num_variadic;
    } else {
      // __shared__[4][32][33] cache is used for column reduction ("4" comes
      // from potential x-tiling).
      return 4 * 32 * 33 * primitive_size * num_variadic;
    }
  } else if (auto tr = GetDescriptionForTiledTransposeEmitter(instr)) {
    // Tile size for transposition.
    int64_t primitive_size =
        ShapeUtil::ByteSizeOfPrimitiveType(instr.shape().element_type());
    int64_t bytes_required = 32 * 33 * primitive_size;
    // If the last dimension is not changed, it becomes part of the tile.
    if (tr->permutation.back() == tr->permutation.size() - 1) {
      bytes_required *= tr->dimensions.back();
    }
    return bytes_required;
  }
  // Other fused expressions for now don't need the shared memory budget.
  return 0;
}

// Returns the number of unnested reductions in the instruction output.
int64_t NumUnnestedReductionsNoCache(const HloInstruction& instr,
                                     const se::DeviceDescription& device_info) {
  if (instr.opcode() == HloOpcode::kReduce &&
      IsReductionFromOrToContiguousDimensions(instr, device_info)) {
    return 1;
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    int64_t sum = 0;
    for (const HloInstruction* hlo :
         instr.fused_instructions_computation()->instructions()) {
      sum += NumUnnestedReductionsNoCache(*hlo, device_info);
    }
    return sum;
  }
  return 0;
}

// Recursive helper for GetFusionRoots below.
void GetFusionRootsRec(const HloInstruction* root,
                       std::vector<const HloInstruction*>& out) {
  if (root->opcode() == HloOpcode::kGetTupleElement &&
      root->operand(0)->opcode() == HloOpcode::kTuple) {
    return GetFusionRootsRec(root->operand(0)->operand(root->tuple_index()),
                             out);
  } else if (root->opcode() == HloOpcode::kGetTupleElement) {
    out.push_back(root->operand(0));
  } else if (root->opcode() == HloOpcode::kTuple) {
    for (int i = 0; i < root->operand_count(); i++) {
      GetFusionRootsRec(root->operand(i), out);
    }
  } else {
    out.push_back(root);
  }
}

bool IsInputFusibleScatter(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kScatter ||
      (instr.opcode() == HloOpcode::kFusion &&
       instr.fusion_kind() == HloInstruction::FusionKind::kInput &&
       instr.fused_expression_root()->opcode() == HloOpcode::kScatter)) {
    return true;
  }
  return false;
}

int64_t SharedMemoryUsage(const HloInstruction& instr, FusionInfoCache* cache,
                          const se::DeviceDescription& device_info) {
  if (!cache) {
    return SharedMemoryUsageNoCache(instr, device_info);
  }
  return cache->GetSharedMemoryUsage(instr);
}

// Codegen'ing unnested reductions requires a lot of registers, so a MOF
// combining many of those runs a high risk of spilling.
constexpr int64_t kMaxUnnestedReductionOutputsPerFusion = 8;

int64_t NumUnnestedReductions(const HloInstruction& instr,
                              FusionInfoCache* cache,
                              const se::DeviceDescription& device_info) {
  if (!cache) {
    return NumUnnestedReductionsNoCache(instr, device_info);
  }
  return cache->GetNumUnnestedReductions(instr);
}

}  // namespace

bool IsPhysicallyTransposing(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kFusion) {
    for (const HloInstruction* fused_instr : instr.fused_instructions()) {
      if (IsPhysicallyTransposing(*fused_instr)) {
        return true;
      }
    }
  }

  // A fusion iterates over its output in physically-contiguous order. This
  // applies "upwards" to operands.  Only an operator that changes an operand's
  // physical layout can create a "bad" memory access pattern.
  return instr.opcode() == HloOpcode::kCopy ||
         (instr.opcode() == HloOpcode::kTranspose &&
          !ShapeUtil::TransposeIsBitcast(instr.operand(0)->shape(),
                                         instr.shape(), instr.dimensions()));
}

bool TransposesMinorDimension(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kFusion:
      return absl::c_any_of(instr->fused_instructions(),
                            TransposesMinorDimension);
    case HloOpcode::kCopy: {
      int64_t first_non_trivial_operand_dim =
          MostMinorNonTrivialDimension(instr->operand(0)->shape()).first;
      int64_t first_non_trivial_output_dim =
          MostMinorNonTrivialDimension(instr->shape()).first;
      return first_non_trivial_operand_dim != first_non_trivial_output_dim;
    }
    case HloOpcode::kTranspose: {
      auto position_in_minor_to_major = InversePermutation(
          instr->operand(0)->shape().layout().minor_to_major());
      int64_t position_of_first_non_trivial_dim =
          MostMinorNonTrivialDimension(instr->operand(0)->shape()).second;
      for (int64_t output_dim : instr->shape().layout().minor_to_major()) {
        if (instr->shape().dimensions()[output_dim] == 1) {
          continue;
        }
        int64_t operand_dim = instr->dimensions().at(output_dim);
        // Check if there is any operand dimension with size > 1 that is more
        // minor than 'operand_dim'
        return position_in_minor_to_major[operand_dim] >
               position_of_first_non_trivial_dim;
      }
      return false;
    }
    default:
      return false;
  }
}

bool IsReduceInputFusion(const HloInstruction& instr,
                         const se::DeviceDescription& device_info) {
  return instr.opcode() == HloOpcode::kFusion &&
         absl::c_any_of(GetFusionRoots(*instr.called_computations()[0]),
                        [&](const HloInstruction* root) {
                          return IsRealReductionHero(
                              *root, FindNonTrivialHero(*root), device_info);
                        });
}

bool IsInputFusibleReduction(const HloInstruction& instr,
                             const se::DeviceDescription& device_info) {
  return IsReduceInputFusion(instr, device_info) ||
         IsReductionFromOrToContiguousDimensions(instr, device_info);
}

std::vector<const HloInstruction*> GetFusionRoots(
    const HloComputation& computation) {
  std::vector<const HloInstruction*> out;
  GetFusionRootsRec(computation.root_instruction(), out);
  return out;
}

int64_t FusionInfoCache::GetSharedMemoryUsage(const HloInstruction& instr) {
  {
    absl::MutexLock lock(&mutex_);
    auto it = shared_memory_usage_.find(&instr);
    if (it != shared_memory_usage_.end()) {
      return it->second;
    }
  }

  // nb: Users are only expected to call cache.Invalidate() on top-level
  // instructions, not instructions inside fusion nodes.  Therefore we can only
  // cache top-level instructions; it would not be valid to pass the cache to
  // SharedMemoryUsageNoCache and use the cache *within* the fusion.
  int64_t shared_memory_usage = SharedMemoryUsageNoCache(instr, device_info_);

  absl::MutexLock lock(&mutex_);
  shared_memory_usage_.emplace(&instr, shared_memory_usage);
  return shared_memory_usage;
}

int64_t FusionInfoCache::GetNumUnnestedReductions(const HloInstruction& instr) {
  {
    absl::MutexLock lock(&mutex_);
    auto it = num_unnested_reductions_.find(&instr);
    if (it != num_unnested_reductions_.end()) {
      return it->second;
    }
  }

  // nb: Users are only expected to call cache.Invalidate() on top-level
  // instructions, not instructions inside fusion nodes.  Therefore we can only
  // cache top-level instructions; it would not be valid to pass the cache to
  // NumUnnestedReductionsNoCache and use the cache *within* the fusion.
  int64_t num_unnested_reductions =
      NumUnnestedReductionsNoCache(instr, device_info_);

  absl::MutexLock lock(&mutex_);
  num_unnested_reductions_.emplace(&instr, num_unnested_reductions);
  return num_unnested_reductions;
}

std::vector<HloComputation*> GetFusibleComputations(
    const HloModule& module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  auto result = module.MakeComputationPostOrder(execution_threads);
  absl::flat_hash_set<const HloComputation*> computations_not_to_fuse;
  for (const auto* computation : result) {
    for (const auto* instr : computation->instructions()) {
      // Don't fuse within called computations, unless they are for control
      // flow.
      if (HloInstruction::MightHaveCalledComputations(instr->opcode()) &&
          instr->opcode() != HloOpcode::kWhile &&
          instr->opcode() != HloOpcode::kConditional &&
          instr->opcode() != HloOpcode::kFusion &&
          instr->opcode() != HloOpcode::kCall) {
        for (auto* called : instr->called_computations()) {
          computations_not_to_fuse.insert(called);
        }
      }
    }
  }
  result.erase(
      std::remove_if(result.begin(), result.end(),
                     [&](HloComputation* computation) {
                       return computation->IsFusionComputation() ||
                              computations_not_to_fuse.contains(computation);
                     }),
      result.end());
  return result;
}

FusionDecision FusionFitsInBudget(const HloInstruction& instr1,
                                  const HloInstruction& instr2,
                                  const se::DeviceDescription& device_info,
                                  bool is_consumer_producer_fusion,
                                  FusionInfoCache* cache) {
  if (SharedMemoryUsage(instr1, cache, device_info) +
          SharedMemoryUsage(instr2, cache, device_info) >
      device_info.shared_memory_per_block()) {
    return FusionDecision::Forbid(
               "shared memory usage would be over the budget of ")
           << device_info.shared_memory_per_block() << "B";
  }

  if (NumUnnestedReductions(instr1, cache, device_info) +
          NumUnnestedReductions(instr2, cache, device_info) >
      kMaxUnnestedReductionOutputsPerFusion) {
    return FusionDecision::Forbid("over ")
           << kMaxUnnestedReductionOutputsPerFusion
           << " unnested reductions in fusion";
  }

  // Compute the number of outputs of the (possibly multi-output) fusion node.
  auto count_outputs = [](const HloInstruction& instr) {
    const Shape& shape = instr.shape();
    return shape.IsTuple() ? shape.tuple_shapes_size() : 1;
  };
  int64_t n_operands = 0;
  int64_t n_outputs = 0;

  // This calculation counts how many inputs and outputs the fused node would
  // have, if we decide to fuse.
  auto compute_n_inputs_n_outputs = [&](const HloInstruction& instr) {
    int64_t n_instr_operands = 0;
    for (auto* instr_operand : instr.operands()) {
      // GetTupleElement and Tuple do not contribute to the number of operands.
      // So don't count as operand.
      if (instr_operand->opcode() == HloOpcode::kTuple ||
          instr_operand->opcode() == HloOpcode::kGetTupleElement) {
        continue;
      }
      // Don't count duplicates.
      if (instr_operand->opcode() == HloOpcode::kFusion) {
        // If the operand is a fusion, the operands of the fusion are
        // potentially shared.
        for (auto* operand : instr_operand->fused_parameters()) {
          auto* actual_operand =
              instr_operand->operand(operand->parameter_number());
          if (!absl::c_linear_search(instr.operands(), actual_operand)) {
            ++n_instr_operands;
          }
        }
      } else {
        ++n_instr_operands;
      }
    }
    n_operands += n_instr_operands;
    n_outputs += count_outputs(instr);
  };
  compute_n_inputs_n_outputs(instr1);
  compute_n_inputs_n_outputs(instr2);

  // Count how many operands will disappear after fusing.
  if (is_consumer_producer_fusion) {
    n_outputs -= count_outputs(instr1);
    n_operands -= instr2.OperandIndices(&instr1).size();
  }

  if (n_operands + n_outputs > MaxOperandsAndOutputsPerFusion()) {
    return FusionDecision::Forbid("Number of operands and outputs exceeds ")
           << MaxOperandsAndOutputsPerFusion();
  }
  return FusionDecision::Allow();
}

FusionDecision CanEmitInputFusedScatter(const HloInstruction& producer,
                                        const HloInstruction& consumer) {
  if (IsInputFusibleScatter(producer)) {
    return FusionDecision::Forbid("do not fuse into the output of scatter");
  }
  if (!IsInputFusibleScatter(consumer)) {
    return FusionDecision::Allow();
  }

  const HloInstruction* inplace_operand;
  if (consumer.opcode() == HloOpcode::kFusion) {
    const HloInstruction* scatter = consumer.fused_expression_root();
    CHECK_EQ(scatter->opcode(), HloOpcode::kScatter);
    CHECK_EQ(scatter->operand(0)->opcode(), HloOpcode::kParameter);
    inplace_operand = consumer.operand(scatter->operand(0)->parameter_number());
  } else {
    inplace_operand = consumer.operand(0);
  }
  if (inplace_operand == &producer) {
    return FusionDecision::Forbid(
        "do not fuse into the in-place operand of scatter");
  }
  if (absl::c_linear_search(producer.operands(), inplace_operand)) {
    return FusionDecision::Forbid(
        "Producer uses the in-place operand of a scatter");
  }
  return FusionDecision::Allow();
}

LaunchDimensionsConfig ComputeLoopFusionConfig(
    const HloFusionAnalysis& analysis) {
  return ComputeLoopFusionConfig(analysis, GetElementShape(analysis));
}

LaunchDimensionsConfig ComputeLoopFusionConfig(
    const HloFusionAnalysis& analysis, const Shape& element_shape) {
  int unroll_factor = 1;
  // Unrolling is good to read large inputs with small elements
  // due to vector loads, but increases the register pressure when one
  // thread has to produce multiple output elements.
  // Therefore for fusions with small outputs prefer to use one thread
  // per output element = no unroll.
  // Call 'small' fusions that use less threads than the GPU has.
  int64_t num_elements = ShapeUtil::ElementsIn(element_shape);
  int64_t n_threads_max = analysis.device_info().threads_per_core_limit() *
                          analysis.device_info().core_count();
  if (num_elements >= n_threads_max &&
      !MayPreventVectorization(analysis.fusion())) {
    unroll_factor = ComputeMaxUnrollFactor(num_elements);
  }
  // CHECK that unroll_factor is a power-of-2, as needed by the logic below.
  CHECK(absl::has_single_bit(static_cast<uint64_t>(unroll_factor)));
  // Ensure a single thread writes to a byte containing multiple values by
  // setting unroll_factor to an appropriate number. Setting unroll_factor is
  // safe even if the new unroll_factor doesn't divide the number of elements,
  // as the parallel loop emitter will insert a bounds check in this case to
  // ensure the out-of-bounds element is not computed and written. Setting
  // unroll_factor is safe even if MayPreventVectorization returns false, as
  // the MayPreventVectorization check is an optimization, not a correctness
  // requirement.
  unroll_factor = std::max(
      unroll_factor,
      CeilOfRatio(8, analysis.input_output_info().smallest_output_dtype_bits));
  CHECK(absl::has_single_bit(static_cast<uint64_t>(unroll_factor)));
  VLOG(2) << "Unroll factor: " << unroll_factor;

  LaunchDimensionsConfig launch_config{unroll_factor};
  return launch_config;
}

}  // namespace zkx::gpu
