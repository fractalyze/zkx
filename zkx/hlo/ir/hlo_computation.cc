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

#include "zkx/hlo/ir/hlo_computation.h"

#include "absl/memory/memory.h"

#include "xla/tsl/platform/status.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_module.h"

namespace zkx {
namespace {

enum class VisitState { kNew = 0, kVisiting = 1, kVisited = 2 };

}  // namespace

class HloComputation::VisitMap {
 public:
  VisitMap() = default;
  explicit VisitMap(int capacity) : size_(capacity) {
    int num_words = (capacity + 31) / 32;
    bits_.resize(num_words);
    bit_ptr_ = bits_.empty() ? nullptr : bits_.data();
  }

  // A handle is a dense index used to identify a particular node.
  using Handle = uint32_t;

  // Returns the current VisitState for the instruction with handle "h"
  VisitState GetState(Handle h) const {
    DCHECK_LT(h, size_);
    uint32_t word = (h / 32);
    uint32_t shift = (h % 32) << 1;
    return static_cast<VisitState>((bit_ptr_[word] >> shift) & 0x3);
  }

  // Sets the VisitState for the instruction with Handle "h" to "new_state"
  void SetState(Handle h, VisitState new_state) {
    DCHECK_LT(h, size_);
    uint32_t word = (h / 32);
    uint32_t shift = (h % 32) << 1;
    uint64_t mask = ~(3ull << shift);
    uint64_t val = static_cast<uint64_t>(new_state);
    bit_ptr_[word] = (bit_ptr_[word] & mask) | (val << shift);
  }

 private:
  // bits_ stores VisitState entries (2 bits per entry, packed 32 entries per
  // 64-bit word)
  absl::InlinedVector<uint64_t, 1> bits_;
  uint64_t* bit_ptr_ = nullptr;  //
  int size_ = 0;  // Number of entries.  bits_ holds at least 2 * this many bits
};

std::unique_ptr<HloComputation> HloComputation::Builder::Build(
    HloInstruction* root_instruction) {
  int parameter_count = 0;
  for (auto& instruction : instructions_) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      parameter_count++;
    }
  }
  // If root_instruction is not specified use the last added instruction.
  HloInstruction* root =
      root_instruction ? root_instruction : last_added_instruction();
  CHECK_NE(nullptr, root);
  return absl::WrapUnique(
      new HloComputation(name_, parameter_count, &instructions_, root));
}

HloComputation::HloComputation(
    const std::string& name, int parameter_count,
    std::vector<std::unique_ptr<HloInstruction>>* instructions,
    HloInstruction* root_instruction)
    : unique_id_(-1),
      root_instruction_(root_instruction),
      instruction_count_(0),
      name_(NameUniquer::GetSanitizedName(name)) {
  param_instructions_.resize(parameter_count, nullptr);
  bool root_found = false;
  for (auto& instruction : *instructions) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      int64_t param_no = instruction->parameter_number();
      CHECK(param_no >= 0 && param_no < parameter_count)
          << "\nERROR: invalid parameter number. Expected [0, "
          << parameter_count << "), got " << param_no;
      CHECK(param_instructions_[param_no] == nullptr)
          << "\nERROR: parameter number " << param_no
          << " already allocated in this computation";
      param_instructions_[param_no] = instruction.get();
    }
    root_found |= instruction.get() == root_instruction_;
    AddInstructionInternal(std::move(instruction));
  }
  CHECK(root_found)
      << "\nERROR: root instruction is not present in computation.";
  root_instruction_->MarkAsRoot();
}

HloComputation::~HloComputation() {
  // TODO(chokobole): Uncomment this. Dependency: FusionInstruction
  // if (FusionInstruction() != nullptr) {
  //   CHECK(FusionInstruction()->fused_instructions_computation() == this);
  //   FusionInstruction()->ClearCalledComputations();
  // }
  if (IsAsyncComputation()) {
    CHECK(async_start_->async_wrapped_computation() == this);
    async_start_->ClearCalledComputations();
  }
  Cleanup();
  for (const auto& i : instructions_) {
    delete i.inst();
  }
}

void HloComputation::SetInstruction(HloInstruction* instruction,
                                    InstructionType type) {
  static_assert(alignof(HloInstruction) == kInstructionTypeMask + 1,
                "HloInstruction should be aligned as a QWORD");

  DCHECK(type != InstructionType::kUnset)
      << "Set instruction must be called with a valid type, not kUnset.";
  DCHECK(instruction_type() == InstructionType::kUnset ||
         instruction_type() == type)
      << "Unexpected instruction type. Current type is "
      << static_cast<int>(instruction_type()) << " and it cannot be reset to "
      << static_cast<int>(type);

  // If `instruction` is nullptr, we need to preserve the existing type.
  if (instruction == nullptr) {
    type = instruction_type();
  }

  instruction_and_type_ =
      reinterpret_cast<uintptr_t>(instruction) | static_cast<uintptr_t>(type);
}

HloInstruction* HloComputation::AddInstruction(
    std::unique_ptr<HloInstruction> instruction, std::string_view new_name) {
  CHECK_NE(instruction->opcode(), HloOpcode::kParameter)
      << "Parameter instructions cannot be added to a computation after "
      << "it has been built";
  if (!new_name.empty()) {
    instruction->SetAndSanitizeName(new_name);
  }
  return AddInstructionInternal(std::move(instruction));
}

HloInstruction* HloComputation::AddInstruction(
    std::unique_ptr<HloInstruction> instruction, const OpMetadata* metadata) {
  if (metadata != nullptr) {
    instruction->set_metadata(*metadata);
  }
  return AddInstruction(std::move(instruction));
}

HloInstruction* HloComputation::AddInstruction(
    std::unique_ptr<HloInstruction> instruction, const OpMetadata* metadata,
    const FrontendAttributes* frontend_attributes) {
  if (frontend_attributes != nullptr) {
    instruction->set_frontend_attributes(*frontend_attributes);
  }
  return AddInstruction(std::move(instruction), metadata);
}

HloInstruction* HloComputation::AddInstructionInternal(
    std::unique_ptr<HloInstruction> instruction) {
  if (parent() != nullptr) {
    instruction->UniquifyName(&parent()->instruction_name_uniquer());
    instruction->SetUniqueId(parent()->NewUniqueInstructionId());
  }
  instruction->set_parent(this);
  HloInstruction* pinst = instruction.release();  // Take ownership
  HloInstructionInfo info;
  info.opcode_ = pinst->opcode();
  info.inst_ = pinst;
  VLOG(2) << "Adding instruction " << pinst << " " << pinst->name()
          << " from computation " << name() << " opcode " << info.opcode();
  uint32_t index = instructions_.size();
  instruction_count_++;
  pinst->index_in_parent_ = index;
  instructions_.push_back(info);
  return pinst;
}

HloInstruction* HloComputation::AddParameter(
    std::unique_ptr<HloInstruction> instruction) {
  CHECK_EQ(instruction->opcode(), HloOpcode::kParameter);
  CHECK(!IsFusionComputation() ||
        FusionInstruction()->operand_count() == param_instructions_.size());
  instruction->set_parent(this);
  param_instructions_.push_back(instruction.get());
  AddInstructionInternal(std::move(instruction));
  return instructions_.back().get();
}

HloInstruction* HloComputation::AddEntryComputationParameter(
    std::unique_ptr<HloInstruction> instruction) {
  CHECK_EQ(instruction->opcode(), HloOpcode::kParameter);
  CHECK_EQ(instruction->parameter_number(), num_parameters());
  CHECK_EQ(parent()->entry_computation(), this);

  HloModuleConfig config = parent()->config();
  config.mutable_entry_computation_layout()->add_parameter_layout(
      ShapeLayout(instruction->shape()));
  parent()->set_config(config);

  instruction->set_parent(this);
  param_instructions_.push_back(instruction.get());
  AddInstructionInternal(std::move(instruction));

  return instructions_.back().get();
}

absl::Status HloComputation::ReplaceEntryComputationParameter(
    int64_t param_no, HloInstruction* old_instruction,
    std::unique_ptr<HloInstruction> instruction) {
  CHECK_GE(param_no, 0);
  CHECK_LT(param_no, param_instructions_.size());
  CHECK_EQ(instruction->opcode(), HloOpcode::kParameter);
  CHECK_EQ(parent()->entry_computation(), this);

  HloModuleConfig config = parent()->config();
  *config.mutable_entry_computation_layout()->mutable_parameter_layout(
      param_no) = ShapeLayout(instruction->shape());
  parent()->set_config(config);

  instruction->set_parent(this);
  param_instructions_[param_no] = instruction.get();
  AddInstructionInternal(std::move(instruction));

  return ForceRemoveInstruction(old_instruction);
}

absl::Status HloComputation::RemoveParameter(int64_t param_no) {
  CHECK_GE(param_no, 0);
  CHECK_LT(param_no, param_instructions_.size());
  HloInstruction* param_instruction = param_instructions_[param_no];
  auto param_instruction_iterator = param_instructions_.begin() + param_no;
  param_instructions_.erase(param_instruction_iterator);
  // Throw removed fused parameter instruction away.
  TF_RETURN_IF_ERROR(ForceRemoveInstruction(param_instruction));

  while (param_no < param_instructions_.size()) {
    param_instruction = param_instructions_[param_no];
    HloInstruction* new_instr = AddInstructionInternal(
        HloInstruction::CreateParameter(param_no, param_instruction->shape(),
                                        absl::StrCat("param_", param_no)));
    TF_RETURN_IF_ERROR(param_instruction->ReplaceAllUsesWith(new_instr));
    param_instructions_[param_no] = new_instr;
    TF_RETURN_IF_ERROR(ForceRemoveInstruction(param_instruction));
    param_no++;
  }

  return absl::OkStatus();
}

HloInstruction* HloComputation::ReplaceParameter(
    int64_t param_no, std::unique_ptr<HloInstruction> instruction) {
  CHECK_GE(param_no, 0);
  CHECK_LT(param_no, param_instructions_.size());
  CHECK_EQ(instruction->opcode(), HloOpcode::kParameter);
  CHECK(!IsFusionComputation() ||
        FusionInstruction()->operand_count() == param_instructions_.size());

  instruction->set_parent(this);
  HloInstruction* new_instruction =
      AddInstructionInternal(std::move(instruction));
  HloInstruction* old_instruction = param_instructions_[param_no];
  TF_CHECK_OK(
      old_instruction->ReplaceAllUsesWithDifferentShape(new_instruction));
  param_instructions_[param_no] = new_instruction;
  TF_CHECK_OK(ForceRemoveInstruction(old_instruction));
  return new_instruction;
}

absl::Status HloComputation::RemoveUnusedParametersFromFusedComputation() {
  return RemoveUnusedParametersImpl(/*allow_non_fusion=*/false);
}

absl::Status HloComputation::RemoveUnusedParametersFromAnyComputation() {
  return RemoveUnusedParametersImpl(/*allow_non_fusion=*/true);
}

absl::Status HloComputation::RemoveUnusedParametersImpl(bool allow_non_fusion) {
  CHECK(allow_non_fusion || IsFusionComputation());
  int64_t removed = 0;
  for (int64_t i = 0; i < param_instructions_.size(); ++i) {
    HloInstruction* param_instruction = param_instructions_[i];
    if (param_instruction->IsDead()) {
      TF_RETURN_IF_ERROR(
          RemoveInstructionImpl(param_instruction, allow_non_fusion));
      ++removed;
      continue;
    }

    if (removed > 0) {
      const int64_t param_no = i - removed;
      HloInstruction* new_instr = AddInstructionInternal(
          HloInstruction::CreateParameter(param_no, param_instruction->shape(),
                                          absl::StrCat("param_", param_no)));
      TF_RETURN_IF_ERROR(param_instruction->ReplaceAllUsesWith(new_instr));
      param_instructions_[param_no] = new_instr;
      TF_RETURN_IF_ERROR(
          RemoveInstructionImpl(param_instruction, allow_non_fusion));
    }
  }
  param_instructions_.resize(param_instructions_.size() - removed);
  return absl::OkStatus();
}

bool HloComputation::IsSafelyRemovable(const HloInstruction* instruction,
                                       bool ignore_control_dependency) {
  // If the instruction has control predecessors or successors then we cannot
  // remove the instruction without violating ordering constraints (added, for
  // example, to avert interference due to buffer aliasing).
  if (!ignore_control_dependency && instruction->HasControlDependencies()) {
    return false;
  }

  if (instruction->opcode() == HloOpcode::kParameter &&
      !IsFusionComputation()) {
    return false;
  }

  return true;
}

bool HloComputation::HasSideEffect() const {
  for (auto* instruction : instructions()) {
    if (instruction->HasSideEffect()) {
      return true;
    }
  }
  return false;
}

bool HloComputation::IsMarkedAsDead(const HloInstruction* inst) {
  return inst->IsMarkedAsDead();
}

absl::Status HloComputation::RemoveInstruction(HloInstruction* instruction) {
  return RemoveInstructionImpl(instruction, /*ignore_safety_check=*/false);
}

absl::Status HloComputation::ForceRemoveInstruction(
    HloInstruction* instruction) {
  return RemoveInstructionImpl(instruction, /*ignore_safety_check=*/true);
}

absl::Status HloComputation::RemoveInstructionImpl(HloInstruction* instruction,
                                                   bool ignore_safety_check) {
  VLOG(2) << "Removing instruction " << instruction << " "
          << instruction->name() << " from computation " << name();
  TF_RET_CHECK(ignore_safety_check || IsSafelyRemovable(instruction))
      << "cannot remove instruction: " << instruction->ToString();
  TF_RET_CHECK(instruction->IsDead()) << "instruction " << instruction->name()
                                      << " is live and cannot be removed";
  TF_RET_CHECK(instruction->control_predecessors().empty())
      << "instruction " << instruction->name()
      << " has control predecessors and cannot be removed";
  TF_RET_CHECK(instruction->control_successors().empty())
      << "instruction " << instruction->name()
      << " has control successors and cannot be removed";

  HloInstructionInfo* info = &instructions_[instruction->index_in_parent_];
  DCHECK_EQ(info->inst(), instruction);
  info->inst()->set_parent(nullptr);
  to_be_deleted_.push_back(info->inst());  // Takes ownership
  to_be_deleted_.back()->DetachFromOperandsAndUsers();
  // Clear all operands to avoid Null operands.
  to_be_deleted_.back()->RemoveAllOperands();
  to_be_deleted_.back()->ClearCalledComputations();
  to_be_deleted_.back()->MarkAsDead();

  // If this instruction is a constant, clear the literal eagerly instead of
  // waiting for the instruction to be deleted in Cleanup(). This greatly
  // reduces the peak heap memory during constant folding.
  if (auto constant = DynCast<HloConstantInstruction>(to_be_deleted_.back())) {
    *constant->mutable_literal() = Literal();
  }
  // TODO(jeff): should we set info->opcode to something?
  info->inst_ =
      nullptr;  // Leave a hole: this is no longer part of "instructions()"
  instruction->index_in_parent_ = ~0u;
  instruction_count_--;
  DCHECK_EQ(instructions_.size() - to_be_deleted_.size(), instruction_count())
      << "instructions_.size(): " << instructions_.size()
      << ", to_be_deleted_.size(): " << to_be_deleted_.size();
  return absl::OkStatus();
}

void HloComputation::Cleanup() {
  if (to_be_deleted_.empty()) return;

  // Given that there are instructions to be deleted, there must be at least one
  // instruction not marked for deletion. Otherwise we have deleted *all*
  // instructions, which is probably a bug.
  DCHECK_GT(instruction_count(), 0);

  // Perform a stable compaction with the erase-remove idiom. We have to open
  // code it (instead of using std::erase(std::remove_if)) because we must
  // update the reverse mapping.
  auto is_marked_for_removal = [](const HloInstructionInfo& info) {
    return info.inst() == nullptr;
  };
  auto marked_it = absl::c_find_if(instructions_, is_marked_for_removal);
  DCHECK(marked_it < instructions_.end());
  for (auto it = marked_it + 1; it < instructions_.end(); ++it) {
    if (is_marked_for_removal(*it)) continue;
    // Update reverse mapping and overwrite the 'marked' entry.
    HloInstruction* unmarked_instruction = it->inst();
    unmarked_instruction->index_in_parent_ =
        std::distance(instructions_.begin(), marked_it);
    *marked_it++ = std::move(*it);
  }

  DCHECK(marked_it < instructions_.end());
  DCHECK_EQ(std::distance(marked_it, instructions_.end()),
            to_be_deleted_.size());
  DCHECK_EQ(instructions_.size() - to_be_deleted_.size(), instruction_count())
      << "instructions_.size(): " << instructions_.size()
      << ", to_be_deleted_.size(): " << to_be_deleted_.size();
  for (HloInstruction* marked_instruction : to_be_deleted_) {
    delete marked_instruction;
  }
  to_be_deleted_.clear();
  instructions_.resize(instruction_count());
}

void HloComputation::set_root_instruction(HloInstruction* new_root_instruction,
                                          bool accept_different_shape) {
  // The shape of the root (ignoring layout) is an invariant of the computation
  // for non-fusion cases.
  if (!IsFusionComputation() && !accept_different_shape) {
    CHECK(ShapeUtil::Compatible(new_root_instruction->shape(),
                                root_instruction_->shape()))
        << new_root_instruction->shape() << " is incompatible with "
        << root_instruction_->shape();
  }
  bool root_found = false;
  for (auto& instruction : instructions_) {
    if (new_root_instruction == instruction.get()) {
      root_found = true;
      break;
    }
  }
  DCHECK(root_found);

  if (parent() && parent()->has_entry_computation() &&
      parent()->entry_computation() == this) {
    if (!Shape::Equal().IgnoreLayout()(new_root_instruction->shape(),
                                       root_instruction_->shape())) {
      // Rebuild input output alias config now that we have a new output
      parent()->input_output_alias_config() =
          HloInputOutputAliasConfig(new_root_instruction->shape());
    }
  }

  // `root_instruction_` can be equal to `new_root_instruction` and so it is
  // important that we call MarkAsNonRoot before calling MarkAsRoot.
  root_instruction_->MarkAsNonRoot();
  new_root_instruction->MarkAsRoot();
  root_instruction_ = new_root_instruction;
}

void HloComputation::ComputeInstructionPostOrder(
    HloInstruction* root, const ChannelDependencies& channel_dependencies,
    VisitMap& visited, std::vector<HloInstruction*>& post_order,
    std::vector<HloInstruction*>* dfs_stack_scratch) const {
  ForEachInstructionPostOrderImpl(
      [&post_order](HloInstruction* hlo) { post_order.push_back(hlo); }, root,
      channel_dependencies, visited, dfs_stack_scratch);
}

void HloComputation::ForEachInstructionPostOrderImpl(
    absl::FunctionRef<void(HloInstruction*)> func, HloInstruction* root,
    const ChannelDependencies& channel_dependencies, VisitMap& visited,
    std::vector<HloInstruction*>* dfs_stack_scratch) const {
  bool has_channel_dependencies = !channel_dependencies.empty();
  auto* dfs_stack = dfs_stack_scratch;
  dfs_stack->clear();

  // Pushes instruction to dfs stack only if it was not already processed.
  auto dfs_stack_push = [&](HloInstruction* instr) {
    VisitState state = visited.GetState(instr->index_in_parent_);
    if (state != VisitState::kVisited) dfs_stack->push_back(instr);
  };

  dfs_stack_push(root);
  while (!dfs_stack->empty()) {
    HloInstruction* current = dfs_stack->back();
    DCHECK_EQ(current->parent(), this)
        << "Instruction " << current->name()
        << " is not in the current computation (" << name() << ").";

    VisitMap::Handle h = current->index_in_parent_;
    VisitState state = visited.GetState(h);
    if (state == VisitState::kNew) {
      visited.SetState(h, VisitState::kVisiting);
    } else {
      dfs_stack->pop_back();
      if (state != VisitState::kVisited) {
        visited.SetState(h, VisitState::kVisited);
        func(current);
      }
      continue;
    }

    // Add channel dependencies.
    // Collectives with the same channel ID must be performed together, as these
    // represent MPMD-partitioned that will later be split into separate modules
    // and the order must be preserved.
    if (has_channel_dependencies && current != root) {
      auto it = channel_dependencies.find(current);
      if (it != channel_dependencies.end()) {
        absl::c_for_each(it->second, dfs_stack_push);
      }
    }

    // Add the operands to the stack in reverse order so the first operand is
    // processed first. This will produce a more natural ordering and a nicer
    // result for things like HLO stringification.
    const HloInstruction::InstructionVector& operands = current->operands();
    absl::c_for_each(tsl::gtl::make_range(operands.rbegin(), operands.rend()),
                     dfs_stack_push);

    // Add control predecessors to the stack.
    absl::c_for_each(current->control_predecessors(), dfs_stack_push);
  }
}

HloComputation::ChannelDependencies HloComputation::ComputeChannelDependencies()
    const {
  if (parent() && parent()->config().has_static_device_assignment() &&
      (parent()->config().static_device_assignment().computation_count() == 1 ||
       parent()->config().use_spmd_partitioning())) {
    return {};
  }

  using Instructions = absl::InlinedVector<HloInstruction*, 1>;
  absl::flat_hash_map<int64_t, Instructions> channel_groups;

  // Create dependencies between partitioned collectives.
  ChannelDependencies dependencies;
  for (const auto& inst : instructions_with_info()) {
    switch (inst.opcode()) {
      case HloOpcode::kAllReduce:
      case HloOpcode::kAllGather:
      case HloOpcode::kAllToAll:
      case HloOpcode::kCollectiveBroadcast:
      case HloOpcode::kCollectivePermute:
      case HloOpcode::kRaggedAllToAll:
      case HloOpcode::kReduceScatter: {
        HloInstruction* instruction = inst.inst();
        std::optional<int64_t> channel_id = instruction->channel_id();
        if (channel_id) {
          Instructions& group = channel_groups[*channel_id];
          for (const HloInstruction* group_inst : group) {
            dependencies[group_inst].push_back(instruction);
          }
          dependencies[instruction] = group;
          group.push_back(instruction);
        }
        break;
      }
      default:
        break;
    }
  }
  return dependencies;
}

std::vector<HloComputation*> HloComputation::MakeEmbeddedComputationsList()
    const {
  absl::flat_hash_set<HloComputation*> visited;
  std::vector<HloComputation*> post_order;
  // The first element of the pair is the currently processed computation, the
  // second is iterator inside the instructions list of the computation that is
  // currently being processed.
  using ComputationIter =
      std::pair<HloComputation*, InstructionList::const_iterator>;
  std::stack<ComputationIter, absl::InlinedVector<ComputationIter, 8>> st;

  // We cannot directly push (this, instructions_.cbegin()) to the stack, as the
  // stack should contain only mutable computations. Also, we don't want to
  // include the computation itself in the list of embedded computations.
  for (const HloInstructionInfo& instruction : instructions_with_info()) {
    using PtrVec = PtrVec<HloComputation*>;
    auto process_called_computations = [&](const PtrVec& called_computations) {
      if (called_computations.empty()) return;
      // Put the called computations in reverse order onto the stack.
      // Otherwise we don't match the recursive enumeration of
      // computations, which processes the first called computation first.
      std::reverse_iterator<PtrVec::const_iterator> i(
          called_computations.end());
      std::reverse_iterator<PtrVec::const_iterator> rend(
          called_computations.begin());
      for (; i != rend; ++i) {
        HloComputation* called_computation = *i;
        if (visited.insert(called_computation).second) {
          st.emplace(called_computation,
                     called_computation->instructions_.cbegin());
        }
      }
    };
    process_called_computations(instruction->called_computations());
    while (!st.empty()) {
      auto& cur = st.top();
      HloComputation* computation = cur.first;
      if (cur.second == computation->instructions_.cend()) {
        st.pop();
        post_order.push_back(computation);
      } else {
        if (cur.second->inst() == nullptr) {
          ++cur.second;
        } else {
          HloOpcode opcode = cur.second->opcode();
          HloInstruction* next_instruction = cur.second->get();
          ++cur.second;
          if (HloInstruction::MightHaveCalledComputations(opcode)) {
            process_called_computations(
                next_instruction->called_computations());
          } else {
            DCHECK(next_instruction->called_computations().empty());
          }
        }
      }
    }
  }

  return post_order;
}

void HloComputation::ForEachInstructionPostOrder(
    absl::FunctionRef<void(HloInstruction*)> func) const {
  VisitMap visited(instructions_.size());
  std::vector<HloInstruction*> dfs_stack_scratch;
  dfs_stack_scratch.reserve(instruction_count());
  auto channel_dependencies = ComputeChannelDependencies();
  for (const auto& instruction : instructions()) {
    if (instruction->users().empty()) {
      ForEachInstructionPostOrderImpl(func, instruction, channel_dependencies,
                                      visited, &dfs_stack_scratch);
    }
  }
}

void HloComputation::Print(Printer* printer,
                           const HloPrintOptions& options) const {
  // Use post-order if order is not specified.
  Print(printer, options, /*instruction_order=*/{});
}

void HloComputation::Print(
    Printer* printer, const HloPrintOptions& options,
    absl::Span<const HloInstruction* const> instruction_order) const {
  if (!instruction_order.empty()) {
    CHECK_EQ(instruction_order.size(), instruction_count());
  }
  const std::string tab(2 * options.indent_amount(), ' ');

  printer->Append(tab);

  if (!options.is_in_nested_computation()) {
    if (options.print_percent()) {
      printer->Append("%");
    }
    if (options.print_ids()) {
      // When print_ids() is false, exclude entry computation's name because it
      // includes and leads to non-deterministic fingerprint.
      printer->Append(name());
      printer->Append(" ");
    }
  }

  if (options.print_program_shape()) {
    ShapeUtil::PrintHumanString(printer,
                                ComputeProgramShape(options.print_ids()));
    printer->Append(" ");
  }
  printer->Append("{\n");

  {
    // Print the instructions in this computation.
    HloPrintOptions new_options =
        HloPrintOptions(options)
            .set_indent_amount(options.indent_amount() + 1)
            .set_is_in_nested_computation(true);

    CanonicalNameMap name_map;
    name_map.Reserve(instruction_count());
    auto print_one = [&](const HloInstruction* instruction) {
      DCHECK_EQ(this, instruction->parent());
      // 2 more spaces than just 'tab' due to indent_amount()+1 above
      printer->Append(tab);
      printer->Append("  ");
      if (instruction == root_instruction_) {
        printer->Append("ROOT ");
      }
      instruction->PrintWithCanonicalNameMap(printer, new_options, &name_map);
      printer->Append("\n");
    };
    // Use post-order if order is not specified.
    if (instruction_order.empty()) {
      ForEachInstructionPostOrder(print_one);
    } else {
      for (const HloInstruction* const instruction : instruction_order) {
        print_one(instruction);
      }
    }
  }

  printer->Append(tab);
  printer->Append("}");
  if (options.print_ids() && !IsMainThread()) {
    // When print_ids() is false, exclude entry computation's thread name
    // because it includes and leads to non-deterministic fingerprint.
    printer->Append(", execution_thread=\"");
    printer->Append(execution_thread());
    printer->Append("\"");
  }
  if (options.print_name_after_closing_brace() && instruction_count() > 5) {
    printer->Append(" // ");
    printer->Append(name());
  }
}

std::string HloComputation::ToString() const {
  return ToString(HloPrintOptions::Default());
}

std::string HloComputation::ToString(const HloPrintOptions& options) const {
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: HloComputation::MakeInstructionPostOrder
  // clang-format on
  // return ToString(options, MakeInstructionPostOrder());
  return "";
}

std::string HloComputation::ToString(
    const HloPrintOptions& options,
    absl::Span<const HloInstruction* const> instruction_order) const {
  StringPrinter printer;
  Print(&printer, options, instruction_order);
  return std::move(printer).ToString();
}

absl::Cord HloComputation::ToCord(
    const HloPrintOptions& options,
    absl::Span<const HloInstruction* const> instruction_order) const {
  CordPrinter printer;
  Print(&printer, options, instruction_order);
  return std::move(printer).ToCord();
}

ProgramShape HloComputation::ComputeProgramShape(bool include_ids) const {
  ProgramShape program_shape;

  for (auto* param_instruction : param_instructions_) {
    *program_shape.add_parameters() = param_instruction->shape();
    *program_shape.add_parameter_names() =
        std::string(PrintName(param_instruction->name(), include_ids));
  }
  *program_shape.mutable_result() = root_instruction_->shape();

  return program_shape;
}

bool HloComputation::EqualInternal(
    const HloComputation& other, bool is_layout_sensitive,
    std::optional<
        absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>>
        computations_comparator,
    bool ignore_channel_id_values, bool ignore_execution_thread) const {
  if (this == &other) {
    return true;
  }
  absl::flat_hash_set<std::pair<const HloInstruction*, const HloInstruction*>>
      visited;
  std::vector<std::pair<const HloInstruction*, const HloInstruction*>> worklist;

  worklist.push_back({root_instruction(), other.root_instruction()});

  while (!worklist.empty()) {
    auto pair = worklist.back();
    worklist.pop_back();

    if (visited.contains(pair)) {
      continue;
    }
    visited.emplace(pair);
    // TODO(b/123082518): Avoid recursively invoking Equal because it may
    // cause a stack overflow with deeply nested subcomputations.
    auto operands_eq = [](const HloInstruction*, const HloInstruction*) {
      return true;
    };

    auto comp_eq = [&](const HloComputation* a, const HloComputation* b) {
      return a->EqualInternal(*b, is_layout_sensitive, computations_comparator,
                              ignore_channel_id_values,
                              ignore_execution_thread);
    };

    bool identical_ignoring_operands =
        ignore_channel_id_values
            ? pair.first->IdenticalIgnoringChannelIdValues(
                  *pair.second, operands_eq,
                  (computations_comparator ? *computations_comparator
                                           : comp_eq),
                  is_layout_sensitive)
            : pair.first->Identical(
                  *pair.second, operands_eq,
                  (computations_comparator ? *computations_comparator
                                           : comp_eq),
                  is_layout_sensitive);
    if (!identical_ignoring_operands) {
      return false;
    }
    for (size_t i = 0; i < pair.first->operands().size(); ++i) {
      worklist.push_back({pair.first->operand(i), pair.second->operand(i)});
    }
  }

  if (!ignore_execution_thread) {
    return execution_thread() == other.execution_thread();
  }
  return true;
}

std::vector<HloInstruction*> HloComputation::CollectUnreachableRoots() const {
  std::vector<HloInstruction*> unreachable_roots;
  for (auto* instruction : instructions()) {
    if (instruction->IsDead() && instruction->control_successors().empty()) {
      unreachable_roots.push_back(instruction);
    }
  }
  VLOG(3) << "Unreachable roots:"
          << absl::StrJoin(unreachable_roots, "\n\t",
                           [](std::string* out, const HloInstruction* hlo) {
                             absl::StrAppend(out, hlo->ToString());
                           });
  return unreachable_roots;
}

void HloComputation::UniquifyName(HloModule* module) {
  UniquifyName(&module->computation_name_uniquer());
}

HloInstruction* HloComputation::GetInstructionWithName(std::string_view name) {
  auto instructions_in_computation = instructions();
  auto it = absl::c_find_if(
      instructions_in_computation,
      [&](HloInstruction* instr) { return instr->name() == name; });
  return it == instructions_in_computation.end() ? nullptr : *it;
}

bool HloComputation::IsEntryComputation() const {
  return parent()->entry_computation() == this;
}

bool HloComputation::CanExpandIntoSingleInstruction() const {
  return absl::c_all_of(
      instructions(), [root = root_instruction()](const HloInstruction* instr) {
        return root == instr || instr->opcode() == HloOpcode::kParameter;
      });
}

}  // namespace zkx
