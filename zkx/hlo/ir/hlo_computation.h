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

#ifndef ZKX_HLO_IR_HLO_COMPUTATION_H_
#define ZKX_HLO_IR_HLO_COMPUTATION_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"

#include "xla/tsl/lib/gtl/iterator_range.h"
#include "xla/tsl/platform/errors.h"
#include "zkx/base/logging.h"
#include "zkx/hlo/ir/dfs_hlo_visitor.h"
#include "zkx/hlo/ir/hlo_clone_context.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/iterator_util.h"
#include "zkx/service/name_uniquer.h"
#include "zkx/shape.h"
#include "zkx/shape_tree.h"
#include "zkx/status_macros.h"

namespace zkx {

class HloModule;

// Describes a computation at the HLO level.
//
// You can think of an HloComputation like a function. It has some inputs
// (parameters) and returns exactly one value (the value of its root node). If
// you want to return multiple values, you can return a tuple.
//
// The instructions inside of a computation do not have an explicit total order.
// Instead, they have a partial order determined by their data and control
// dependencies.
//
// An HloModule contains one "entry computation" -- this is like main() in a C
// program. Every other computation inside of a module is attached to one or
// more HloInstructions, as a "nested computation". For example, the kMap
// instruction has a nested computation and "applies" it to every element of its
// input, elementwise. (That is, the input [x, y, z] is transformed to [f(x),
// f(y), f(z)].)
class HloComputation {
 public:
  // Used by instructions_.
  using InstructionList = std::vector<HloInstructionInfo>;

  // Builder class for HloComputation.
  class Builder {
   public:
    explicit Builder(std::string_view name) : name_(name) {}
    Builder(Builder&& b) = default;
    virtual ~Builder() = default;

    // Build and return an HloComputation. The parameter root_instruction
    // specifies the already-added instruction to use as the root. If
    // `root_instruction` is nullptr then use the last added instruction as the
    // root.
    std::unique_ptr<HloComputation> Build(
        HloInstruction* root_instruction = nullptr);

    // Add the instruction to be part of this computation.
    // If the new instruction is derived from another one,
    // you probably want to do
    // `original_inst->AddInstruction(new_inst)` instead.
    virtual HloInstruction* AddInstruction(
        std::unique_ptr<HloInstruction> instruction) {
      auto* added_instruction = instruction.get();
      instructions_.push_back(std::move(instruction));
      return added_instruction;
    }

    HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction,
                                   std::optional<std::string_view> new_name) {
      instruction->SetAndSanitizeName(new_name.value());
      return AddInstruction(std::move(instruction));
    }

    absl::StatusOr<HloInstruction*> AddParameter(
        std::unique_ptr<HloInstruction> parameter) {
      if (!parameter_numbers_.insert(parameter->parameter_number()).second) {
        return absl::InternalError(absl::StrFormat(
            "Duplicate parameter number %d", parameter->parameter_number()));
      }
      return AddInstruction(std::move(parameter));
    }

    absl::Status ForEachInstruction(
        absl::FunctionRef<absl::Status(const HloInstruction*)> func) const {
      for (const auto& instruction : instructions_) {
        TF_RETURN_IF_ERROR(func(instruction.get()));
      }
      return absl::OkStatus();
    }

    HloInstruction* last_added_instruction() const {
      return instructions_.empty() ? nullptr : instructions_.back().get();
    }

   private:
    const std::string name_;
    std::vector<std::unique_ptr<HloInstruction>> instructions_;
    absl::flat_hash_set<int> parameter_numbers_;

    Builder(const Builder&) = delete;
    Builder& operator=(const Builder&) = delete;
  };

  // // Helper class to automatically set the OpMetadata for every instruction
  // // added to a computation.
  // class MetadataBuilder {
  //  public:
  //   MetadataBuilder(HloComputation* computation, const OpMetadata& metadata)
  //       : computation_(computation), metadata_(metadata) {}

  //   HloInstruction* AddInstruction(
  //       std::unique_ptr<HloInstruction> instruction) {
  //     instruction->set_metadata(metadata_);
  //     return computation_->AddInstruction(std::move(instruction));
  //   }

  //  private:
  //   HloComputation* computation_;
  //   OpMetadata metadata_;
  // };

  ~HloComputation();

  enum class InstructionType : uint8_t {
    kUnset,
    // This computation is a fusion computation. A fusion computation ordinarily
    // also has a non-null instruction. However, if a fusion instruction
    // is removed during compilation, the fusion computation becomes
    // unreachable, and its instruction is set to null. We still need to regard
    // such computations as fusion computations for HLO scheduling purposes.
    kFusion,
    // This computation is a custom-call computation.
    kCustomCall,
    // This computation is a collective computation.
    kCollective,
    // This computation is a while body computation.
    kWhile,
    // This computation is a conditional branch computation.
    kConditional,
    // Last Value for range checking.
    kLast = kConditional,
  };
  static constexpr uintptr_t kInstructionTypeMask = 0b111;
  static_assert(static_cast<int>(InstructionType::kUnset) == 0,
                "kUnset must be 0.");

  InstructionType instruction_type() const {
    return static_cast<InstructionType>(instruction_and_type_ &
                                        kInstructionTypeMask);
  }

  HloInstruction* instruction() const {
    DCHECK(instruction_type() <= InstructionType::kLast);
    return reinterpret_cast<HloInstruction*>(instruction_and_type_ &
                                             ~kInstructionTypeMask);
  }
  // Add an instruction to the computation. The computation takes ownership of
  // the instruction.
  HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction,
                                 std::string_view new_name = "");

  HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction,
                                 const OpMetadata* metadata);

  HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction,
                                 const OpMetadata* metadata,
                                 const FrontendAttributes* frontend_attributes);

  // Replace the old parameter at index param_no with
  // `instruction`. Updates uses and root instruction. Removes old
  // instruction from computation. No check is done on the shape.
  HloInstruction* ReplaceParameter(int64_t param_no,
                                   std::unique_ptr<HloInstruction> instruction);

  // Remove the param_no'th parameter from the computation.
  // Note this is only applicable to the computation for the fusion
  // instruction.
  absl::Status RemoveParameter(int64_t param_no);

  // Remove unused parameters from the computation.
  // Note this is only applicable to the computation for the fusion
  // instruction.
  absl::Status RemoveUnusedParametersFromFusedComputation();

  // Remove unused parameters from the computation. Unlike
  // RemoveUnusedParametersFromFusedComputation, this function can be used
  // to remove parameters from non-fusion computations.
  absl::Status RemoveUnusedParametersFromAnyComputation();

  // Adds a new parameter instruction to a fusion computation.
  //
  // This should be a new parameter. Instruction will be appended to parameters
  // and inserted to the instruction list.
  HloInstruction* AddParameter(std::unique_ptr<HloInstruction> instruction);

  // Adds a new parameter instruction to the entry computation and update
  // the parent module config to reflect the change.
  //
  // This should be a new parameter. Instruction will be appended to parameters
  // and inserted to the instruction list.
  HloInstruction* AddEntryComputationParameter(
      std::unique_ptr<HloInstruction> instruction);

  // Replaces an old parameter with a new parameter. Adds the new parameter
  // instruction to the entry computation.  Updates users instruction.
  absl::Status ReplaceEntryComputationParameter(
      int64_t param_no, HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> instruction);

  // Remove an instruction from the computation. The instruction must have no
  // users. This call does not yet deallocate the instruction, but marks it as
  // deleted, so that the next call to Cleanup() will deallocate it. If the
  // instruction is a constant, its literal is cleared.
  absl::Status RemoveInstruction(HloInstruction* instruction);

  // Removes an instruction from the computation. The instruction must have no
  // users. The instruction will be removed even if it is marked as not
  // removable. This call does not yet deallocate the instruction, but marks it
  // as deleted, so that the next call to Cleanup() will deallocate it. If the
  // instruction is a constant, its literal is cleared.
  absl::Status ForceRemoveInstruction(HloInstruction* instruction);

  // Set the root of the computation to the given instruction. The instruction
  // must have already been added to the computation. In addition it must have
  // the same shape as the result of the computation for non fusion
  // computations, except if accept_different_shape is set to true.
  void set_root_instruction(HloInstruction* new_root_instruction,
                            bool accept_different_shape = false);

  // Return the root instruction of the computation. The root instruction is the
  // instruction which produces the output of the computation.
  HloInstruction* root_instruction() const { return root_instruction_; }

  // Returns the number of parameters for this computation.
  int64_t num_parameters() const { return param_instructions_.size(); }

  // Returns the parameter instruction for the given parameter number.
  HloInstruction* parameter_instruction(int64_t param_no) const {
    CHECK_GE(param_no, 0);
    CHECK_LT(param_no, static_cast<int64_t>(param_instructions_.size()))
        << "Computation " << name() << " has no parameter number " << param_no;
    return param_instructions_[param_no];
  }

  const HloInstruction::InstructionVector& parameter_instructions() const {
    return param_instructions_;
  }

  std::string_view name() const { return name_; }

  // Sets the string identifier for this computation. Name will be sanitized to
  // match the regexp "[a-zA-Z_][a-zA-Z0-9_.-]*".
  //
  // See also HloModule::SetAndUniquifyComputationName(), which does this plus
  // UniqufyName().
  void SetAndSanitizeName(std::string_view name) {
    name_ = NameUniquer::GetSanitizedName(name);
  }

  // Use the given NameUniquer to select a unique name for the computation based
  // on the computation's existing name.
  //
  // See also HloModule::SetAndUniquifyComputationName(), which does this plus
  // SetAndSanitizeName().
  void UniquifyName(NameUniquer* name_uniquer) {
    name_ = name_uniquer->GetUniqueName(name_);
  }

  // Use the given `module` to select a unique name for this computation based
  // on computation's existing name.
  void UniquifyName(HloModule* module);

  // Prints a string representation of the computation.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  void Print(Printer* printer) const {
    return Print(printer, HloPrintOptions::Default());
  }
  void Print(Printer* printer, const HloPrintOptions& options) const;
  void Print(Printer* printer, const HloPrintOptions& options,
             absl::Span<const HloInstruction* const> instruction_order) const;

  // Return a string representation of the computation.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  std::string ToString() const;
  std::string ToString(const HloPrintOptions& options) const;

  // Overload which accepts an order to emit the instructions in.
  std::string ToString(
      const HloPrintOptions& options,
      absl::Span<const HloInstruction* const> instruction_order) const;

  // Returns a Cord representation of the computation.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)

  // Overload which accepts an order to emit the instructions in.
  absl::Cord ToCord(
      const HloPrintOptions& options,
      absl::Span<const HloInstruction* const> instruction_order) const;

  // Returns a serialized representation of this computation.
  // TODO(chokobole): Uncomment this. Dependency: HloComputationProto
  // HloComputationProto ToProto() const;

  // Creates a computation from the given proto. Arguments:
  //
  //   proto: the proto to convert from.
  //   computation_map: a map from computation id to HloComputation*. This map
  //     must contain all computations which the newly constructed computation
  //     calls.
  // TODO(chokobole): Uncomment this. Dependency: HloComputationProto
  // static absl::StatusOr<std::unique_ptr<HloComputation>> CreateFromProto(
  //     const HloComputationProto& proto,
  //     const absl::flat_hash_map<int64_t, HloComputation*>& computation_map,
  //     bool prohibit_empty_literal = true);

  using InstructionSequence = tsl::gtl::iterator_range<
      UnwrappingIterator<HloInstructionList::iterator>>;

  using ConstInstructionSequence = tsl::gtl::iterator_range<
      UnwrappingIterator<HloInstructionList::const_iterator>>;

  // Gets the instructions in this computation.
  //
  // The returned type is a range of HloInstruction*s, so you can iterate over
  // it using a range-based for loop in the natural way:
  //
  //   for (HloInstruction* instr : computation->instructions()) { ... }
  //

  tsl::gtl::iterator_range<HloInstructionUnwrappingConstIterator> instructions()
      const {
    const int end = instructions_.size();
    return {HloInstructionUnwrappingConstIterator(
                HloInstructionConstIterator(&instructions_, 0, end)),
            HloInstructionUnwrappingConstIterator(
                HloInstructionConstIterator(&instructions_, end, end))};
  }
  tsl::gtl::iterator_range<HloInstructionUnwrappingIterator> instructions() {
    const int end = instructions_.size();
    return {HloInstructionUnwrappingIterator(
                HloInstructionIterator(&instructions_, 0, end)),
            HloInstructionUnwrappingIterator(
                HloInstructionIterator(&instructions_, end, end))};
  }
  tsl::gtl::iterator_range<HloInstructionIterator> instructions_with_info() {
    const int end = instructions_.size();
    return {HloInstructionIterator(&instructions_, 0, end),
            HloInstructionIterator(&instructions_, end, end)};
  }
  tsl::gtl::iterator_range<HloInstructionConstIterator> instructions_with_info()
      const {
    const int end = instructions_.size();
    return {HloInstructionConstIterator(&instructions_, 0, end),
            HloInstructionConstIterator(&instructions_, end, end)};
  }

  using ChannelDependencies =
      absl::flat_hash_map<const HloInstruction*,
                          absl::InlinedVector<HloInstruction*, 1>>;

  // Calls `func` with each instruction in the computation in post-order.
  void ForEachInstructionPostOrder(
      absl::FunctionRef<void(HloInstruction*)> func) const;

  int64_t instruction_count() const { return instruction_count_; }

  // Creates and returns a list of the embedded computations called by this
  // computation. This includes all embedded computations called directly or
  // transitively. The embedded computations are sorted such that if computation
  // A calls computation B (eg, via a map instruction) then A will appear after
  // B in the list.
  std::vector<HloComputation*> MakeEmbeddedComputationsList() const;

  // Computes and returns the ProgramShape of this computation (shape of
  // parameters and result with layout).
  ProgramShape ComputeProgramShape(bool include_ids = true) const;

  // Return whether `*this` and `other` are functionally equivalent.
  bool Equal(
      const HloComputation& other, bool is_layout_sensitive,
      std::optional<
          absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>>
          computations_comparator = std::nullopt) const {
    return EqualInternal(other, is_layout_sensitive, computations_comparator,
                         /*ignore_channel_id_values=*/false,
                         /*ignore_execution_thread=*/false);
  }

  // Same as Equal() but ignores channel ID value mismatches on instructions, as
  // long as the two instructions both have channel IDs or neither has a channel
  // ID.
  bool EqualIgnoringChannelIdValues(
      const HloComputation& other, bool is_layout_sensitive,
      std::optional<
          absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>>
          computations_comparator = std::nullopt) const {
    return EqualInternal(other, is_layout_sensitive, computations_comparator,
                         /*ignore_channel_id_values=*/true,
                         /*ignore_execution_thread=*/false);
  }

  bool EqualIgnoringExecutionThread(
      const HloComputation& other, bool is_layout_sensitive,
      bool ignore_channel_id_values,
      std::optional<
          absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>>
          computations_comparator = std::nullopt) const {
    return EqualInternal(other, is_layout_sensitive, computations_comparator,
                         ignore_channel_id_values,
                         /*ignore_execution_thread=*/true);
  }

  // Return whether `*this` and `other` are functionally equivalent.
  bool operator==(const HloComputation& other) const {
    return Equal(other, true);
  }
  bool operator!=(const HloComputation& other) const {
    return !(*this == other);
  }

  // Set/get the module containing this computation.
  void set_parent(HloModule* module) { parent_ = module; }
  const HloModule* parent() const { return parent_; }
  HloModule* parent() { return parent_; }

  // Returns true if the given instruction can be removed from the computation.
  // Parameter instructions cannot be removed without violating invariants of
  // the HLO computation with the exception of fusion computation. A parameter
  // instruction is removable for a fusion computation.
  //
  // Note that IsSafelyRemovable() is a necessary condition to remove an
  // instruction rather than a sufficient condition. For example, instructions
  // with side-effect (e.g., Send, Infeed) may be removed from a computation,
  // but the transformation must guarantee the invariants relevant to the
  // instructions still hold (e.g., Send and Recv must be removed together to
  // make each channel complete).
  bool IsSafelyRemovable(const HloInstruction* instruction,
                         bool ignore_control_dependency = false);

  // Returns a map from an instruction to the group of instructions associated
  // with the same channel. These instructions will be considered as a single
  // node for dependency purposes.
  // RecvDone ops will map to the corresponding Send op.
  // Cross-partition collectives will map to every other instruction with the
  // same channel ID (it doesn't map to itself).
  ChannelDependencies ComputeChannelDependencies() const;

  // Returns true if this computation has a side effect. A computation has a
  // side effect if it contains one or more instructions with a side effect.
  bool HasSideEffect() const;

  // Returns if this computation is a fusion computation.
  // Do not use this method to determine if fusion_instruction_ != nullptr.
  // Instead, directly do: FusionInstruction() != nullptr
  bool IsFusionComputation() const {
    return instruction_type() == InstructionType::kFusion;
  }

  // Returns if this computation is the entry computation of the module.
  bool IsEntryComputation() const;

  // Returns the owning fusion instruction, or nullptr if this is not a fusion
  // computation.
  HloInstruction* FusionInstruction() const {
    return instruction_type() == InstructionType::kFusion ? instruction()
                                                          : nullptr;
  }
  void SetFusionInstruction(HloInstruction* fusion_instruction) {
    SetInstruction(fusion_instruction, InstructionType::kFusion);
  }

  // Returns if this computation is a custom-call computation.
  bool IsCustomCallComputation() const {
    return instruction_type() == InstructionType::kCustomCall;
  }

  // Returns the owning custom call instruction, or nullptr if this is not a
  // custom call computation.
  HloInstruction* CustomCallInstruction() const {
    return instruction_type() == InstructionType::kCustomCall ? instruction()
                                                              : nullptr;
  }
  void SetCustomCallInstruction(HloInstruction* custom_call_instruction) {
    SetInstruction(custom_call_instruction, InstructionType::kCustomCall);
  }

  // Returns if this computation is a to_apply region of a collective.
  bool IsCollectiveCalledComputation() const {
    return instruction_type() == InstructionType::kCollective;
  }

  // Returns the owning collective call instruction, or nullptr if this is not a
  // collective call computation.
  HloInstruction* CollectiveCallInstruction() const {
    return instruction_type() == InstructionType::kCollective ? instruction()
                                                              : nullptr;
  }

  void SetCollectiveCallInstruction(
      HloInstruction* collective_call_instruction) {
    SetInstruction(collective_call_instruction, InstructionType::kCollective);
  }

  // Returns if this computation is an async computation.
  bool IsAsyncComputation() const { return async_start_ != nullptr; }

  // Returns true if this computation only contains send/recv instructions.
  bool OnlyContainsSendRecv() {
    for (const HloInstruction* instruction : this->instructions()) {
      if (!HloPredicateIsOp<HloOpcode::kSend, HloOpcode::kRecv,
                            HloOpcode::kBitcast, HloOpcode::kParameter,
                            HloOpcode::kTuple>(instruction)) {
        return false;
      }
    }
    return true;
  }

  // Returns the owning async instruction. It's nullptr if this is not an
  // async computation.
  HloInstruction* AsyncStart() const { return async_start_; }

  void AddAsyncStart(HloInstruction* async_instruction) {
    // TODO: Add instruction type for async instructions.
    CHECK(instruction_type() == InstructionType::kUnset);
    CHECK(async_instruction->opcode() == HloOpcode::kAsyncStart);
    async_start_ = async_instruction;
  }

  void RemoveAsyncStart() { async_start_ = nullptr; }

  // Clear the unique ID of the computation so that it can be re-assigned,
  // such as for the purpose of compacting the unique IDs.
  void ClearUniqueIdInternal() { unique_id_ = -1; }

  // The id of this computation should be unique within the module.
  void SetUniqueId(int64_t id) {
    CHECK_EQ(unique_id_, -1);
    CHECK_GE(id, 0);
    unique_id_ = id;
  }

  // Returns the instruction in this computation that has name `name`. Returns
  // null if there is no such computation.
  HloInstruction* GetInstructionWithName(std::string_view name);

  int64_t unique_id() const { return unique_id_; }

  void SetExecutionThread(std::string_view execution_thread) {
    execution_thread_ = std::string(execution_thread);
  }

  std::string_view execution_thread() const { return execution_thread_; }
  // Returns true if this computation is annotated on "main" execution thread.
  bool IsMainThread() const {
    return execution_thread_ == HloInstruction::kMainExecutionThread;
  }

  // Deallocates instructions that are marked by "RemoveInstruction" and
  // compacts the instructions_ vector by removing the deleted instructions'
  // entries (a.k.a. tombstones).
  // This two-stage clean up process is designed such that HloPass can have
  // stable internal pointers to HloInstructions while we create and remove
  // HloInstructions in a pass.
  // Note: the removal operation is stable because some users depend on it.
  void Cleanup();

  // Returns true if a given instruction is marked dead in this computation.
  bool IsMarkedAsDead(const HloInstruction* inst);

  // Returns true iff this computation can be inlined as a single instruction.
  bool CanExpandIntoSingleInstruction() const;

 private:
  explicit HloComputation(
      const std::string& name, int parameter_count,
      std::vector<std::unique_ptr<HloInstruction>>* instructions,
      HloInstruction* root_instruction);

  // Internal helper for adding instructions.
  HloInstruction* AddInstructionInternal(
      std::unique_ptr<HloInstruction> instruction);

  // Internal helper for comparison with different options.
  bool EqualInternal(
      const HloComputation& other, bool is_layout_sensitive,
      std::optional<
          absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>>
          computations_comparator,
      bool ignore_channel_id_values, bool ignore_execution_thread) const;

  // Internal helper to collect unreachable roots.
  std::vector<HloInstruction*> CollectUnreachableRoots() const;

  class VisitMap;
  void ComputeInstructionPostOrder(
      HloInstruction* root, const ChannelDependencies& channel_dependencies,
      VisitMap& visited, std::vector<HloInstruction*>& post_order,
      std::vector<HloInstruction*>* dfs_stack_scratch) const;

  void ForEachInstructionPostOrderImpl(
      absl::FunctionRef<void(HloInstruction*)> func, HloInstruction* root,
      const ChannelDependencies& channel_dependencies, VisitMap& visited,
      std::vector<HloInstruction*>* dfs_stack_scratch) const;

  absl::Status RemoveUnusedParametersImpl(bool allow_non_fusion);

  absl::Status RemoveInstructionImpl(HloInstruction* instruction,
                                     bool ignore_safety_check);

  void SetInstruction(HloInstruction* instruction, InstructionType type);

  int64_t unique_id_;
  HloInstruction* root_instruction_;

  // Module containing this computation.
  HloModule* parent_ = nullptr;

  // Contains HloInstruction* and its type.
  // The respective type in the least significant three bits.
  uintptr_t instruction_and_type_ = 0;

  // If this computation is an async computation, this field points to the
  // first async instruction (async-start) in the asynchronous op chain that
  // calls this computation.
  // Otherwise, this is empty.
  HloInstruction* async_start_ = nullptr;

  HloInstruction::InstructionVector param_instructions_;

  // Store instructions in std::vector as they can be added and removed
  // arbitrarily and we want a stable iteration order.
  // For the reverse mapping we use HloInstruction::index_in_parent_.
  //
  // Note: removals from this vector must be stable because some users depend
  // on it. See the Cleanup() method for details on the two-stage removal
  // process.
  HloInstructionList instructions_;

  // Number of not-marked-for-deletion entries in instructions_.
  int64_t instruction_count_;

  // Removed instructions are moved into to_be_deleted_ first and then
  // deallocated when Cleanup is called.
  PtrVec<HloInstruction*> to_be_deleted_;

  // Execution thread of this computation. By default, it's main thread.
  std::string execution_thread_ = HloInstruction::kMainExecutionThread;

  std::string name_;

  HloComputation(const HloComputation&) = delete;
  HloComputation& operator=(const HloComputation&) = delete;
};

}  // namespace zkx

#endif  // ZKX_HLO_IR_HLO_COMPUTATION_H_
