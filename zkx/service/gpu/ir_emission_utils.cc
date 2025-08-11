/* Copyright 2017 The OpenXLA Authors.

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

#include "zkx/service/gpu/ir_emission_utils.h"

#include "zkx/primitive_util.h"

namespace zkx::gpu {

std::optional<TransposeDescription> GetDescriptionForTiledTransposeEmitter(
    const HloInstruction& hero) {
  if (hero.opcode() != HloOpcode::kTranspose) {
    return std::nullopt;
  }

  // We can assume that TransposeDimensionGrouper pass has run, so no need to
  // call GetNormalizedLogicalTransposeShape here.
  absl::InlinedVector<int64_t, 3> permutation(hero.dimensions().begin(),
                                              hero.dimensions().end());
  // A real transpose needs at least 2 transpose dimensions.
  if (permutation.size() < 2) {
    return std::nullopt;
  }
  absl::InlinedVector<int64_t, 3> dimensions(hero.shape().dimensions().begin(),
                                             hero.shape().dimensions().end());
  int64_t operand_most_minor_dim = hero.operand(0)->shape().dimensions().back();
  if (permutation.back() == dimensions.size() - 1) {
    operand_most_minor_dim =
        hero.operand(0)->shape().dimensions(dimensions.size() - 2);
    auto byte_width = primitive_util::ByteWidth(hero.shape().element_type());
    if (byte_width * dimensions.back() <= kMaxBytesInMostMinorDimension &&
        byte_width * dimensions.back() *
                std::min(operand_most_minor_dim,
                         dimensions[dimensions.size() - 2]) >=
            kMinDimensionToTransposeTiled) {
      return TransposeDescription{&hero, dimensions, permutation};
    }
  } else if ((operand_most_minor_dim >= kMinDimensionToTransposeTiled &&
              dimensions.back() >= kMinDimensionToTransposeTiled) ||
             (operand_most_minor_dim >= kMinDimensionToTransposeTiled2 &&
              dimensions.back() >= kMinDimensionToTransposeTiled2 &&
              operand_most_minor_dim * dimensions.back() >=
                  kMinTotalDimensionsToTransposeTiled)) {
    return TransposeDescription{&hero, dimensions, permutation};
  }
  return std::nullopt;
}

}  // namespace zkx::gpu
