/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/service/transpose_folding.h"

#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"

#include "zkx/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/status_macros.h"
#include "zkx/util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {
namespace {

bool IsNonIdentityTranspose(const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kTranspose) {
    for (int dim = 0; dim < instruction->dimensions().size(); ++dim) {
      if (dim != instruction->dimensions(dim)) {
        return true;
      }
    }
  }
  return false;
}

void TransposeDims(google::protobuf::RepeatedField<int64_t>& dims,
                   absl::Span<const int64_t> transpose_dims) {
  for (auto& dim : dims) {
    dim = transpose_dims[dim];
  }
}

using InstructionOperandsPair =
    std::pair<HloInstruction*, TransposeFolding::OperandIndices>;

// Folds the operands of `dot` that are foldable transposes.
absl::Status FoldTransposeIntoDot(InstructionOperandsPair& pair) {
  HloInstruction* dot = pair.first;

  DotDimensionNumbers new_dot_dims = dot->dot_dimension_numbers();
  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);

  for (int64_t operand_index : pair.second) {
    if (operand_index == 0) {
      TransposeDims(*new_dot_dims.mutable_lhs_contracting_dimensions(),
                    lhs->dimensions());
      TransposeDims(*new_dot_dims.mutable_lhs_batch_dimensions(),
                    lhs->dimensions());
      lhs = lhs->mutable_operand(0);
    } else {
      CHECK_EQ(operand_index, 1);
      TransposeDims(*new_dot_dims.mutable_rhs_contracting_dimensions(),
                    rhs->dimensions());
      TransposeDims(*new_dot_dims.mutable_rhs_batch_dimensions(),
                    rhs->dimensions());
      rhs = rhs->mutable_operand(0);
    }
  }
  HloInstruction* new_dot = dot->parent()->AddInstruction(
      HloInstruction::CreateDot(dot->shape(), lhs, rhs, new_dot_dims));
  dot->SetupDerivedInstruction(new_dot);
  return dot->parent()->ReplaceInstruction(dot, new_dot);
}

}  // namespace

TransposeFolding::TransposeFolding(
    CanFoldTransposeOperand dot_can_fold_transpose_operand)
    : dot_can_fold_transpose_operand_(
          std::move(dot_can_fold_transpose_operand)) {}

absl::StatusOr<bool> TransposeFolding::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  // Modifying the graph while traversing is dangerous, so we find all folding
  // opportunities before actually folding them.
  std::vector<InstructionOperandsPair> foldable_dots;
  std::vector<InstructionOperandsPair> foldable_convolutions;

  FunctionVisitor visit_fn([this, &foldable_dots](HloInstruction* instruction) {
    if (instruction->opcode() == HloOpcode::kDot) {
      // Don't fold dots with a 1D operand.
      if ((instruction->operand(0)->shape().rank() < 2) ||
          (instruction->operand(1)->shape().rank() < 2)) {
        return absl::OkStatus();
      }

      OperandIndices operand_indices;
      for (int64_t i = 0; i < 2; ++i) {
        if (!IsNonIdentityTranspose(instruction->operand(i))) {
          continue;
        }

        TF_ASSIGN_OR_RETURN(bool can_fold_operand,
                            dot_can_fold_transpose_operand_(*instruction, i));

        if (can_fold_operand) {
          operand_indices.push_back(i);
        }
      }

      if (!operand_indices.empty()) {
        foldable_dots.emplace_back(instruction, operand_indices);
      }
    }

    return absl::OkStatus();
  });

  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    TF_RETURN_IF_ERROR(comp->Accept(&visit_fn));
  }

  bool changed = false;
  for (InstructionOperandsPair& pair : foldable_dots) {
    TF_RETURN_IF_ERROR(FoldTransposeIntoDot(pair));
    changed = true;
  }
  return changed;
}

// static
absl::StatusOr<bool> TransposeFolding::IsRowColumnTransposeDotOperand(
    const HloInstruction& dot, int64_t operand_idx) {
  TF_RET_CHECK(dot.opcode() == HloOpcode::kDot);
  TF_RET_CHECK(dot.operand_count() > operand_idx);

  const HloInstruction& transpose = *dot.operand(operand_idx);
  TF_RET_CHECK(transpose.opcode() == HloOpcode::kTranspose);

  const DotDimensionNumbers& dot_dims = dot.dot_dimension_numbers();

  auto batch_dims = (operand_idx == 0) ? dot_dims.lhs_batch_dimensions()
                                       : dot_dims.rhs_batch_dimensions();

  auto contracting_dims = (operand_idx == 0)
                              ? dot_dims.lhs_contracting_dimensions()
                              : dot_dims.rhs_contracting_dimensions();

  return (batch_dims.size() == transpose.shape().rank() - 2) &&
         (contracting_dims.size() == 1) &&
         absl::c_all_of(batch_dims, [&](int64_t dim) {
           return transpose.dimensions(dim) == dim;
         });
}

}  // namespace zkx
