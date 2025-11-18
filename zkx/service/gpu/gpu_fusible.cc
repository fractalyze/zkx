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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"

#include "zkx/hlo/utils/hlo_traversal.h"
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

}  // namespace

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
