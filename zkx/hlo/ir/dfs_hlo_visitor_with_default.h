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

#ifndef ZKX_HLO_IR_DFS_HLO_VISITOR_WITH_DEFAULT_H_
#define ZKX_HLO_IR_DFS_HLO_VISITOR_WITH_DEFAULT_H_

#include <functional>
#include <memory>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/ir/dfs_hlo_visitor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"

namespace zkx {

// DfsHloVisitor with default action based on the HloInstruction being visited.
// Users should not use this class directly, but use the type aliases
// DfsHloVisitorWithDefault/ConstDfsHloVisitorWithDefault instead.
//
// Do *not* add an override to this class if the opcode is covered by
// HandleElementwiseUnary/Binary. These opcode handlers dispatch to
// HandleElementwiseUnary/Binary in DfsHloVisitorBase. Adding such a handler
// here will break passes which rely on the HandleElementwiseUnary/Binary
// handling these opcodes.
template <typename HloInstructionPtr>
class DfsHloVisitorWithDefaultBase
    : public DfsHloVisitorBase<HloInstructionPtr> {
 public:
  DfsHloVisitorWithDefaultBase() = default;
  ~DfsHloVisitorWithDefaultBase() override = default;

  // Default action performed on HloInstruction.
  virtual absl::Status DefaultAction(HloInstructionPtr hlo_instruction) = 0;

  absl::Status HandleElementwiseUnary(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleElementwiseBinary(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }

  absl::Status HandleAllGather(HloInstructionPtr crs) override {
    return DefaultAction(crs);
  }
  absl::Status HandleAllGatherStart(HloInstructionPtr crs) override {
    return DefaultAction(crs);
  }
  absl::Status HandleAllGatherDone(HloInstructionPtr crs) override {
    return DefaultAction(crs);
  }
  absl::Status HandleAllReduce(HloInstructionPtr crs) override {
    return DefaultAction(crs);
  }
  absl::Status HandleReduceScatter(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleAllReduceStart(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleAllReduceDone(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleAllToAll(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleRaggedAllToAll(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleCollectiveBroadcast(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleCollectivePermute(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleCollectivePermuteStart(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleCollectivePermuteDone(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleReplicaId(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandlePartitionId(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleConstant(HloInstructionPtr constant) override {
    return DefaultAction(constant);
  }
  absl::Status HandleParameter(HloInstructionPtr parameter) override {
    return DefaultAction(parameter);
  }

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  absl::Status FinishVisit(HloInstructionPtr /*root*/) override {
    return absl::OkStatus();
  }

 private:
  DfsHloVisitorWithDefaultBase(const DfsHloVisitorWithDefaultBase&) = delete;
  DfsHloVisitorWithDefaultBase& operator=(const DfsHloVisitorWithDefaultBase&) =
      delete;
};

// Users should use one of these two type aliases, which are the only two valid
// instantiations of DfsHloVisitorWithDefaultBase.
using DfsHloVisitorWithDefault = DfsHloVisitorWithDefaultBase<HloInstruction*>;
using ConstDfsHloVisitorWithDefault =
    DfsHloVisitorWithDefaultBase<const HloInstruction*>;

// A common base class for visitors performing rewriting operation.
//
// Subclasses call ReplaceWithNewInstruction and ReplaceInstruction while
// visiting.
class DfsHloRewriteVisitor : public DfsHloVisitorWithDefault {
 public:
  // Runs a visitor on the module and returns whether the module has changed.
  absl::StatusOr<bool> RunOnModule(
      HloModule* module,
      const absl::flat_hash_set<std::string_view>& execution_threads = {}) {
    absl::Status status;
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      // TODO(chokobole): Uncomment this. Dependency: HloComputation::Accept
      // status = computation->Accept(this);
      static_cast<void>(computation);
      if (ABSL_PREDICT_FALSE(!status.ok())) return status;
    }
    return changed();
  }

  // Default visitor action is to do nothing and return OK.
  absl::Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return absl::OkStatus();
  }

  bool changed() const { return changed_; }

 protected:
  // Replaces the existing HLO instruction old_instruction, with
  // new_instruction, and marks the optimizer status as changed.
  // Returns the absl::Status representing the result of the replace operation.
  absl::Status ReplaceWithNewInstruction(
      HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> new_instruction) {
    VLOG(3) << "Replacing instruction:" << "\n  old: "
            << old_instruction->ToString()
            << "\n  new: " << new_instruction->ToString();
    // clang-format off
    // TODO(chokobole): Uncomment this. Dependency: HloComputation::ReplaceWithNewInstruction
    // clang-format on
    // absl::Status status =
    // old_instruction->parent()->ReplaceWithNewInstruction(
    //     old_instruction, std::move(new_instruction));
    absl::Status status = absl::UnimplementedError("...");
    if (ABSL_PREDICT_TRUE(status.ok())) {
      changed_ = true;
    }
    return status;
  }

  // Replaces the existing HLO instruction old_instruction, with
  // new_instruction, and marks the optimizer status as changed.
  // Returns the absl::Status representing the result of the replace operation.
  absl::StatusOr<bool> ReplaceInstruction(HloInstruction* old_instruction,
                                          HloInstruction* new_instruction,
                                          bool preserve_sharding) {
    VLOG(3) << "Replacing instruction:" << "\n  old: "
            << old_instruction->ToString()
            << "\n  new: " << new_instruction->ToString();
    // clang-format off
    // TODO(chokobole): Uncomment this. Dependency: HloComputation::ReplaceInstruction
    // clang-format on
    // absl::StatusOr<bool> changed_or =
    //     old_instruction->parent()->ReplaceInstruction(
    //         old_instruction, new_instruction, preserve_sharding,
    //         /*relay_control_dependency=*/true);
    absl::StatusOr<bool> changed_or = absl::UnimplementedError("...");
    if (ABSL_PREDICT_TRUE(changed_or.ok())) {
      changed_ |= changed_or.value();
    }
    return changed_or;
  }

  absl::Status ReplaceInstruction(HloInstruction* old_instruction,
                                  HloInstruction* new_instruction) {
    absl::StatusOr<bool> changed_or =
        ReplaceInstruction(old_instruction, new_instruction,
                           /*preserve_sharding=*/false);
    if (ABSL_PREDICT_TRUE(changed_or.ok())) {
      DCHECK(changed_or.value());
    }
    return changed_or.status();
  }

  // Mark the computation as having changed.
  void MarkAsChanged() { changed_ = true; }
  void MarkAsMaybeChanged(bool changed) { changed_ |= changed; }

 private:
  bool changed_ = false;
};

// (Const)FunctionVisitor lets you transform an
// std::function<absl::Status((const) HloInstruction*)> into a
// (Const)DfsHloVisitor.
//
// This is useful if you have code that needs to handle visitors in the form of
// both std::function and DfsHloVisitor.  You can wrap the function in a
// FunctionVisitor and then treat it like any other DfsHloVisitor.
template <typename HloInstructionPtr>
class FunctionVisitorBase
    : public DfsHloVisitorWithDefaultBase<HloInstructionPtr> {
 public:
  explicit FunctionVisitorBase(
      std::function<absl::Status(HloInstructionPtr)> visitor_func)
      : visitor_func_(std::move(visitor_func)) {}

  absl::Status DefaultAction(HloInstructionPtr hlo_instruction) override {
    return visitor_func_(hlo_instruction);
  }

 private:
  FunctionVisitorBase(const FunctionVisitorBase&) = delete;
  FunctionVisitorBase& operator=(const FunctionVisitorBase&) = delete;

  std::function<absl::Status(HloInstructionPtr)> visitor_func_;
};

using FunctionVisitor = FunctionVisitorBase<HloInstruction*>;
using ConstFunctionVisitor = FunctionVisitorBase<const HloInstruction*>;

}  // namespace zkx

#endif  // ZKX_HLO_IR_DFS_HLO_VISITOR_WITH_DEFAULT_H_
