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

#ifndef ZKX_SERVICE_TRANSPOSE_FOLDING_H_
#define ZKX_SERVICE_TRANSPOSE_FOLDING_H_

#include <functional>

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/pass/hlo_pass_interface.h"

namespace zkx {

// HLO pass that folds transpose operators into Dot operators, where the Dot
// operator is implemented by a GEMM kernel that can transpose its inputs.
class TransposeFolding : public HloModulePass {
 public:
  using OperandIndices = std::vector<int64_t>;

  using CanFoldTransposeOperand = std::function<absl::StatusOr<bool>(
      const HloInstruction&, int64_t /*operand_idx*/)>;

  // Helper function to explicitly not fold transposes.
  static OperandIndices NeverFoldTranspose(const HloInstruction&,
                                           const OperandIndices&) {
    return {};
  }

  // Helper function to always fold transposes.
  static OperandIndices AlwaysFoldTranspose(const HloInstruction&,
                                            const OperandIndices& ids) {
    return ids;
  }

  // `dot_can_fold_transpose_operand` returns whether the dot operation can fold
  // in the given transpose operand.
  explicit TransposeFolding(
      CanFoldTransposeOperand dot_can_fold_transpose_operand =
          IsRowColumnTransposeDotOperand);
  std::string_view name() const override { return "transpose-folding"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<std::string_view>& execution_threads) override;

  static absl::StatusOr<bool> IsRowColumnTransposeDotOperand(
      const HloInstruction& dot, int64_t operand_idx);

 private:
  CanFoldTransposeOperand dot_can_fold_transpose_operand_;
};

}  // namespace zkx

#endif  // ZKX_SERVICE_TRANSPOSE_FOLDING_H_
