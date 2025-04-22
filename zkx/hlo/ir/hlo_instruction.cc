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

#include "zkx/hlo/ir/hlo_instruction.h"

#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"

#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_op_metadata.h"
#include "zkx/shape_util.h"

namespace zkx {

// static
const HloInstruction::Rare* const HloInstruction::kEmptyRare =
    new HloInstruction::Rare;

namespace {
// Specialization for erasing from PtrVec<T>.
template <typename T>
absl::Status EraseElementFromVector(PtrVec<T>* container, T value) {
  // absl::c_find returns a const_iterator which does not seem to work on
  // gcc 4.8.4, and this breaks the ubuntu/xla_gpu build bot.
  auto it = std::find(container->begin(), container->end(), value);
  TF_RET_CHECK(it != container->end());
  container->erase(it);
  return absl::OkStatus();
}
}  // namespace

void HloInstruction::Users::Clear() {
  users_.clear();
  user_map_.reset(nullptr);
  DCHECK(CheckInvariants());
}

bool HloInstruction::Users::Contains(const HloInstruction* instruction) const {
  if (user_map_ == nullptr) {
    return std::find(users_.begin(), users_.end(), instruction) != users_.end();
  } else {
    return user_map_->contains(instruction);
  }
}

void HloInstruction::Users::AddUser(HloInstruction* user) {
  if (!Contains(user)) {
    // Create hash table if user list is large.
    if (user_map_ == nullptr && users_.size() >= kMapThreshold) {
      user_map_ =
          std::make_unique<absl::flat_hash_map<const HloInstruction*, int64_t>>(
              users_.size());
      RebuildMap();
      DCHECK(CheckInvariants());
    }

    if (user_map_ != nullptr) {
      user_map_->emplace(user, users_.size());
    }
    users_.push_back(user);
    DCHECK(CheckInvariants());
  }
}

int64_t HloInstruction::Users::UserId(HloInstruction* user) {
  if (user_map_ == nullptr) {
    auto it = std::find(users_.begin(), users_.end(), user);
    CHECK(it != users_.end());
    return it - users_.begin();
  } else {
    auto result = user_map_->find(user);
    CHECK(result != user_map_->end());
    return result->second;
  }
}

void HloInstruction::Users::MaybeRemoveUser(HloInstruction* user) {
  if (Contains(user)) {
    RemoveUser(user);
    DCHECK(CheckInvariants());
  }
}

void HloInstruction::Users::RemoveUser(HloInstruction* user) {
  const int64_t index = UserId(user);
  CHECK_EQ(users_[index], user);

  // Move the last user into the position of the removed user.
  HloInstruction* last = users_.back();

  // Update map if allocated.
  if (user_map_ != nullptr) {
    (*user_map_)[last] = index;
    user_map_->erase(user);
  }

  // Replace found user with last slot from the vector.
  users_[index] = last;
  users_.pop_back();

  DCHECK(CheckInvariants());
}

void HloInstruction::Users::SortInstructionUsers(
    const MappedPtrContainerSorter<HloInstruction>::MapPtrFn& map_fn,
    const Users& sorted_instruction_users) {
  using Sorter = MappedPtrContainerSorter<HloInstruction>;
  auto status = Sorter::Sort(map_fn, Sorter::IndexAfterMappedElementsFn(),
                             sorted_instruction_users.users_, users_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to sort instruction users: " << status;
  }
  if (user_map_ != nullptr) {
    user_map_->clear();
    RebuildMap();
  }
  DCHECK(CheckInvariants());
}

void HloInstruction::Users::RebuildMap() {
  for (uint64_t i = 0; i < users_.size(); ++i) {
    (*user_map_)[users_[i]] = i;
  }
}

bool HloInstruction::Users::CheckInvariants() {
  if (user_map_ != nullptr) {
    // Avoid quadratic behavior by doing a quick and dirty check on
    // size instead of actually comparing mapped indices.
    CHECK_EQ(users_.size(), user_map_->size());
  }
  return true;
}

void HloInstruction::AppendComputation(HloComputation* computation) {
  // In .cc file since PtrVec<T*>::push_back() wants to check the alignment
  // of T and hlo_instruction.h does not include hlo_computation.h.
  mutable_rare()->called_computations.push_back(computation);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateParameter(
    int64_t parameter_number, const Shape& shape, std::string_view name) {
  return std::make_unique<HloParameterInstruction>(parameter_number, shape,
                                                   name);
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateConstant(
    Literal literal) {
  return std::make_unique<HloConstantInstruction>(std::move(literal));
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateNary(
    const Shape& shape, HloOpcode opcode,
    absl::Span<HloInstruction* const> operands) {
  if (opcode == HloOpcode::kCopy) {
    // It is impossible to copy an opaque shape, we don't know how big it is.
    CHECK(!shape.IsOpaque());
  }
  auto instruction = absl::WrapUnique(new HloInstruction(opcode, shape));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  return instruction;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateUnary(
    const Shape& shape, HloOpcode opcode, HloInstruction* operand) {
  // Only certain opcodes are supported with CreateUnary: opcodes of unary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kBitcast:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyDone:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kNegate:
      return CreateNary(shape, opcode, {operand});
    default:
      LOG(FATAL) << "Invalid unary instruction opcode " << opcode;
  }
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateBinary(
    const Shape& shape, HloOpcode opcode, HloInstruction* lhs,
    HloInstruction* rhs) {
  // Only certain opcodes are supported with CreateBinary: opcodes of binary
  // instructions with no auxiliary fields.
  switch (opcode) {
    case HloOpcode::kAdd:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kSubtract:
      break;
    default:
      LOG(FATAL) << "Invalid binary instruction opcode " << opcode;
  }
  return CreateNary(shape, opcode, {lhs, rhs});
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateAfterAll(
    absl::Span<HloInstruction* const> operands) {
  CHECK(!operands.empty());
  auto instruction = absl::WrapUnique(
      new HloInstruction(HloOpcode::kAfterAll, ShapeUtil::MakeTokenShape()));
  for (auto operand : operands) {
    instruction->AppendOperand(operand);
  }
  return instruction;
}

// static
std::unique_ptr<HloInstruction> HloInstruction::CreateToken() {
  return absl::WrapUnique(
      new HloInstruction(HloOpcode::kAfterAll, ShapeUtil::MakeTokenShape()));
}

void HloInstruction::set_single_sharding(const HloSharding& sharding) {
  CHECK(!sharding.IsTuple()) << sharding;
  if (shape().IsTuple()) {
    set_sharding(HloSharding::Tuple(sharding.GetAsShapeTree(shape())));
  } else {
    set_sharding(sharding);
  }
}

void HloInstruction::SetupDerivedInstruction(
    HloInstruction* derived_instruction) const {
  if (sharding_ != nullptr &&
      ShapeUtil::CompatibleKind(shape_, derived_instruction->shape())) {
    // Only copy sharding if the tuple tree shape of the two instruction is
    // compatible because copying it between differently shaped instructions
    // can produce invalid shardings.
    derived_instruction->set_sharding(*sharding_);
  } else if (!ShapeUtil::CompatibleKind(shape_, derived_instruction->shape())) {
    derived_instruction->clear_sharding();
  }
  derived_instruction->set_metadata(*metadata_);
  if (has_rare()) {
    derived_instruction->set_frontend_attributes(frontend_attributes());
    derived_instruction->set_statistics_viz(statistics_viz());
  } else if (derived_instruction->has_rare()) {
    derived_instruction->mutable_rare()->frontend_attributes.Clear();
    derived_instruction->mutable_rare()->statistics_viz.Clear();
  }
  // If the derived instruction has the same opcode as current, then the backend
  // config is also applicable (only if derived instruction doesn't have its own
  // backend config which might be different from the original one).
  if (opcode() == derived_instruction->opcode() && has_backend_config() &&
      !derived_instruction->has_backend_config()) {
    derived_instruction->CopyBackendConfigFrom(this);
  }
}

bool HloInstruction::HasSideEffectNoRecurse() const {
  switch (opcode_) {
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCollectivePermuteDone:
      return true;

    case HloOpcode::kAllToAll:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
      if (Cast<HloCollectiveInstruction>(this)->constrain_layout()) {
        return true;
      }
      [[fallthrough]];
    case HloOpcode::kCollectivePermute:
      // Collective instructions with channel_id are side effecting only if
      // they are used in non-spmd context.
      return Cast<HloChannelInstruction>(this)->channel_id().has_value() &&
             !GetModule()->config().use_spmd_partitioning();

    // TODO(chokobole): Uncomment this. Dependency: HloCustomCallInstruction
    // case HloOpcode::kCustomCall:
    //   return Cast<HloCustomCallInstruction>(this)
    //       ->custom_call_has_side_effect();
    default:
      return false;
  }
}

bool HloInstruction::HasSideEffect() const {
  if (HasSideEffectNoRecurse()) {
    return true;
  }
  // Check if any of the called computations has a side effect.
  for (const auto& computation : called_computations()) {
    if (computation->HasSideEffect()) {
      return true;
    }
  }
  return false;
}

// static
bool HloInstruction::IsThreadIncluded(
    std::string_view execution_thread,
    const absl::flat_hash_set<std::string_view>& execution_threads_set) {
  return execution_threads_set.empty() ||
         execution_threads_set.contains(execution_thread);
}

void HloInstruction::DetachFromOperandsAndUsers() {
  if (cleaned_up_) {
    return;
  }
  cleaned_up_ = true;
  // Detach from operands. An instruction may be repeated as an operand. To
  // avoid calling RemoveUser twice on the same operand, check before remove.
  for (int64_t operand_num = 0; operand_num < operand_count(); ++operand_num) {
    HloInstruction* operand = operands_[operand_num];
    if (operand == nullptr) {
      continue;
    }
    operand->users_.MaybeRemoveUser(this);
    operands_[operand_num] = nullptr;
  }

  // Update users. Set `nullptr` to the corresponding operand slot for users.
  for (auto& user : this->users()) {
    for (int i = 0; i < user->operand_count(); ++i) {
      if (user->operands_[i] == this) {
        user->operands_[i] = nullptr;
      }
    }
  }
}

void HloInstruction::AddSuffixToInstructionName(const std::string_view suffix) {
  // If an instruction is cloned multiple times avoid names like
  // foo.suffix.suffix.suffix. Instead of repeating the suffix add a numeric
  // suffix. Specifically, the clone of foo.suffix is named foo.suffix2, the
  // clone of foo.suffix2 is named foo.suffix3 and so on.
  const std::string dot_suffix = absl::StrCat(".", suffix);
  size_t index = name().rfind(dot_suffix);
  if (index == std::string::npos) {
    // Existing name does not include ".suffix".
    this->name_ = absl::StrCat(name(), dot_suffix);
  } else {
    // Existing name includes ".suffix". Determine if substring after
    // ".suffix" is numeric and should be replaced with an incremented number.
    auto after_suffix = name().substr(index + dot_suffix.size());
    if (after_suffix.empty()) {
      // Existing name ends in ".suffix". New name should end in ".suffix2".
      this->name_ = absl::StrCat(name(), "2");
    } else {
      // If names ends with .suffix[0-9]+ then replace with a suffix with the
      // numeric value incremented.
      int64_t numeric_suffix;
      if (absl::SimpleAtoi(after_suffix, &numeric_suffix)) {
        this->name_ = absl::StrCat(name().substr(0, index), dot_suffix,
                                   numeric_suffix + 1);
      } else {
        // Substring after ".suffix" is non-numeric.
        this->name_ = absl::StrCat(name(), dot_suffix);
      }
    }
  }
}

std::unique_ptr<HloInstruction> HloInstruction::CloneWithNewOperands(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return CloneWithNewOperands(shape, new_operands, "", context);
}

std::unique_ptr<HloInstruction> HloInstruction::CloneWithNewOperands(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    const std::string& suffix, HloCloneContext* context) const {
  VLOG(3) << "CloneWithNewOperands:\n  " << ToString();
  VLOG(3) << "  new operands:";
  for (const HloInstruction* new_operand : new_operands) {
    VLOG(3) << "    %" << new_operand->name();
  }

  std::unique_ptr<HloInstruction> clone;
  // Explicitly call the factory for the instruction type. This is more robust
  // in the face of code changes than copying fields explicitly. This also
  // properly sets the user fields of the operands.
  switch (opcode_) {
    // Ops migrated to subclasses.
    // TODO(b/80131774): Remove this switch when migration is complete.
    case HloOpcode::kCompare:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReverse:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReduce:
    case HloOpcode::kTranspose:
    case HloOpcode::kBroadcast:
    case HloOpcode::kReshape:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kMap:
    case HloOpcode::kSlice:
    case HloOpcode::kConstant:
    case HloOpcode::kFusion:
    case HloOpcode::kParameter:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllToAll:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kScatter:
    case HloOpcode::kDot:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kDomain:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
      clone = CloneWithNewOperandsImpl(shape, new_operands, context);
      break;
    // Unary ops.
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kBitcast:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCopy:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kCopyDone:
    case HloOpcode::kNegate:
      CHECK_EQ(new_operands.size(), 1);
      clone = CreateUnary(shape, opcode_, new_operands[0]);
      break;
    // Binary ops.
    case HloOpcode::kAdd:
    case HloOpcode::kDivide:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kPower:
      CHECK_EQ(new_operands.size(), 2);
      clone = CreateBinary(shape, opcode_, new_operands[0], new_operands[1]);
      break;
    case HloOpcode::kAfterAll:
      if (new_operands.empty()) {
        clone = CreateToken();
      } else {
        clone = CreateAfterAll(new_operands);
      }
      break;
    default:
      CHECK(0) << "Unsupported opcode: " << opcode_;
  }
  // SetupDerivedInstruction will setup the precision_config_ field.
  SetupDerivedInstruction(clone.get());
  clone->set_parent(parent_);
  clone->backend_config_ = BackendConfigWrapper(backend_config_);
  // The new instruction's name will be uniquified when it's added to a
  // computation.
  clone->SetAndSanitizeName(name());
  if (context != nullptr) {
    context->MapInstruction(this, clone.get());
    // clang-format off
    // TODO(chokobole): Uncomment this. Dependency: HloModule::DeepCloneComputation
    // clang-format on
    // clone->ReplaceCalledComputations([&](HloComputation* callee) {
    //   return callee->parent() != context->module()
    //              ? context->module()->DeepCloneComputation(callee, context)
    //              : callee;
    // });
    // TODO(chokobole): Uncomment this. Dependency: HloModule::while_body
    // if (opcode() == HloOpcode::kWhile) {
    //   clone->while_body()->SetWhileCallInstruction(clone.get());
    // }
  }

  if (!suffix.empty()) {
    clone->AddSuffixToInstructionName(suffix);
  }
  return clone;
}

std::unique_ptr<HloInstruction> HloInstruction::CloneWithNewShape(
    const Shape& shape, const std::string& suffix,
    HloCloneContext* context) const {
  std::unique_ptr<HloInstruction> clone =
      CloneWithNewOperands(shape, operands_, context);
  if (suffix.empty()) {
    clone->name_.assign(name().begin(), name().end());
  } else {
    clone->AddSuffixToInstructionName(suffix);
  }
  return clone;
}

std::unique_ptr<HloInstruction> HloInstruction::Clone(
    const std::string& suffix, HloCloneContext* context) const {
  std::unique_ptr<HloInstruction> clone =
      CloneWithNewShape(shape_, suffix, context);
  return clone;
}

HloInstruction::InstructionVector HloInstruction::unique_operands() const {
  InstructionVector unique;
  absl::flat_hash_set<const HloInstruction*> seen;
  for (HloInstruction* operand : operands()) {
    if (seen.insert(operand).second) {
      unique.push_back(operand);
    }
  }
  return unique;
}

absl::Status HloInstruction::AddControlDependencyTo(
    HloInstruction* instruction) {
  TF_RET_CHECK(instruction->parent() == parent());
  if (!absl::c_linear_search(control_successors(), instruction)) {
    mutable_rare()->control_successors.push_back(instruction);
    TF_RET_CHECK(!absl::c_linear_search(
        instruction->rare()->control_predecessors, this));
    instruction->mutable_rare()->control_predecessors.push_back(this);
  }
  return absl::OkStatus();
}

absl::Status HloInstruction::RemoveControlDependencyTo(
    HloInstruction* instruction) {
  TF_RET_CHECK(instruction->parent() == parent());
  if (has_rare()) {
    TF_RETURN_IF_ERROR(EraseElementFromVector(
        &mutable_rare()->control_successors, instruction));
  }
  if (instruction->has_rare()) {
    TF_RETURN_IF_ERROR(EraseElementFromVector(
        &instruction->mutable_rare()->control_predecessors, this));
  }
  return absl::OkStatus();
}

absl::Status HloInstruction::DropAllControlDeps() {
  if (has_rare()) {
    for (auto* ctrl_succ : rare()->control_successors) {
      TF_RETURN_IF_ERROR(EraseElementFromVector(
          &ctrl_succ->mutable_rare()->control_predecessors, this));
    }
    for (auto* ctrl_pred : rare()->control_predecessors) {
      TF_RETURN_IF_ERROR(EraseElementFromVector(
          &ctrl_pred->mutable_rare()->control_successors, this));
    }
    Rare* r = mutable_rare();
    r->control_successors.clear();
    r->control_predecessors.clear();
  }
  return absl::OkStatus();
}

absl::Status HloInstruction::SafelyDropAllControlDependencies() {
  // Add all pairs of transitive dependencies from predecessors to successors.
  if (has_rare()) {
    for (HloInstruction* predecessor : rare()->control_predecessors) {
      for (HloInstruction* successor : rare()->control_successors) {
        TF_RETURN_IF_ERROR(predecessor->AddControlDependencyTo(successor));
      }
    }
  }
  TF_RETURN_IF_ERROR(DropAllControlDeps());
  return absl::OkStatus();
}

bool HloInstruction::HasControlDependencies() const {
  const Rare* r = rare();
  return (!r->control_predecessors.empty() || !r->control_successors.empty());
}

absl::Status HloInstruction::CopyAllControlDepsTo(HloInstruction* start,
                                                  HloInstruction* end) const {
  for (auto* ctrl_pred : control_predecessors()) {
    TF_RETURN_IF_ERROR(ctrl_pred->AddControlDependencyTo(start));
  }
  for (auto* ctrl_succ : control_successors()) {
    TF_RETURN_IF_ERROR(end->AddControlDependencyTo(ctrl_succ));
  }
  return absl::OkStatus();
}

bool HloInstruction::IdenticalInternal(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloInstruction*, const HloInstruction*)>
        eq_operands,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations,
    bool layout_sensitive, bool sharding_sensitive,
    bool ignore_channel_id_values,
    bool ignore_commutative_operand_order) const {
  // An instruction is always identical to itself.
  if (this == &other) {
    return true;
  }

  // Identical instruction must have the same opcode, shape, shardings and
  // identical operands.
  if (opcode() != other.opcode()) {
    return false;
  }
  if (!(layout_sensitive ? ShapeUtil::Equal(shape(), other.shape())
                         : ShapeUtil::Compatible(shape(), other.shape()))) {
    return false;
  }
  if (sharding_sensitive && has_sharding() && other.has_sharding() &&
      sharding() != other.sharding()) {
    return false;
  }
  if (operands().size() != other.operands().size()) {
    return false;
  }

  // Check that operands are equal.
  //
  // Use an explicit loop rather than ContainerEquals, because copying around
  // std::functions may be too expensive in some cases.
  if (ignore_commutative_operand_order &&
      HloOpcodeIsBinaryCommutative(opcode())) {
    CHECK_EQ(operand_count(), 2);
    if (!(eq_operands(operand(0), other.operand(0)) &&
          eq_operands(operand(1), other.operand(1))) &&
        !(eq_operands(operand(0), other.operand(1)) &&
          eq_operands(operand(1), other.operand(0)))) {
      return false;
    }
  } else {
    for (size_t i = 0; i < operands().size(); ++i) {
      if (!eq_operands(operand(i), other.operand(i))) {
        return false;
      }
    }
  }

  if (backend_config_ != other.backend_config_) {
    return false;
  }

  if (ignore_channel_id_values) {
    if (auto channel_inst = DynCast<HloChannelInstruction>(this)) {
      return channel_inst->IdenticalSlowPathIgnoringChannelIdValues(
          other, eq_computations);
    }
  }
  return IdenticalSlowPath(other, eq_computations);
}

void HloInstruction::AppendOperand(HloInstruction* operand) {
  if (operand->parent() != nullptr) {
    DCHECK(!operand->parent()->IsMarkedAsDead(operand))
        << "Operand " << operand->name() << " is already marked dead";
  }
  operands_.push_back(operand);
  operand->AddUser(this);
}

void HloInstruction::AppendOperands(
    absl::Span<HloInstruction* const> operands) {
  for (HloInstruction* operand : operands) {
    HloInstruction::AppendOperand(operand);
  }
}

bool HloInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  // Perform opcode specific checks.
  switch (opcode()) {
    // The result of these instructions only depend upon their opcode and
    // operands.
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAdd:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCopyDone:
    case HloOpcode::kDivide:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kPartitionId:
    case HloOpcode::kPower:
    case HloOpcode::kReshape:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kReplicaId:
    case HloOpcode::kSelect:
    case HloOpcode::kSubtract:
    case HloOpcode::kTuple:
      return true;

    // This opcode has complex or special behavior so just return false.
    case HloOpcode::kAfterAll:
    case HloOpcode::kAddDependency:
      return false;

      // Remaining instructions with special values.
    case HloOpcode::kCall:
      return eq_computations(to_apply(), other.to_apply());
    case HloOpcode::kConditional:
      // TODO(chokobole): Uncomment this. Dependency: branch_count
      //   for (int j = 0; j < branch_count(); ++j) {
      //     if (!eq_computations(branch_computation(j),
      //                          other.branch_computation(j))) {
      //       return false;
      //     }
      //   }
      //   return true;
      return false;
    case HloOpcode::kWhile:
      // TODO(chokobole): Uncomment this. Dependency: while_body,
      // while_condition
      //   return (eq_computations(while_body(), other.while_body()) &&
      //           eq_computations(while_condition(), other.while_condition()));
      return false;

    // Ops migrated to subclasses should never come to this line.
    // TODO(b/80131774): Remove this switch when migration is complete.
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kCompare:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReverse:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReduce:
    case HloOpcode::kTranspose:
    case HloOpcode::kBroadcast:
    case HloOpcode::kMap:
    case HloOpcode::kSlice:
    case HloOpcode::kConstant:
    case HloOpcode::kFusion:
    case HloOpcode::kParameter:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kGather:
    case HloOpcode::kScatter:
    case HloOpcode::kDot:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kDomain:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kSetDimensionSize:
      LOG(FATAL) << "Base class impl called for opcode with subclass: "
                 << opcode();
  }
  return false;
}

absl::Status HloInstruction::ReplaceUseWith(HloInstruction* user,
                                            HloInstruction* new_producer) {
  TF_RET_CHECK(ShapeUtil::Compatible(shape(), new_producer->shape()))
      << "this shape: " << ShapeUtil::HumanString(shape())
      << ", replacement shape: "
      << ShapeUtil::HumanString(new_producer->shape());
  return ReplaceUseWithDifferentShape(user, new_producer);
}

absl::Status HloInstruction::ReplaceUseWithDifferentShape(
    HloInstruction* user, HloInstruction* new_producer) {
  VLOG(3) << "Replacing uses of " << name() << " in " << user->name()
          << " with " << new_producer->name();

  RemoveUser(user);

  TF_RET_CHECK(absl::c_count(user->operands_, this) >= 0);
  std::replace(user->operands_.begin(), user->operands_.end(), this,
               new_producer);
  new_producer->AddUser(user);
  // Custom fusions may not be able to handle deduplicated operands.
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: HloOpcode::kFusion, HloFusionInstruction
  // clang-format on
  // if (user->opcode() == HloOpcode::kFusion) {
  //   TF_RETURN_IF_ERROR(
  //       Cast<HloFusionInstruction>(user)->DeduplicateFusionOperands());
  // }
  return absl::OkStatus();
}

absl::Status HloInstruction::ReplaceUseWith(HloInstruction* user,
                                            int operand_number,
                                            HloInstruction* new_producer) {
  TF_RET_CHECK(ShapeUtil::Compatible(shape(), new_producer->shape()))
      << "this shape: " << ShapeUtil::HumanString(shape())
      << ", replacement shape: "
      << ShapeUtil::HumanString(new_producer->shape());
  return ReplaceUseWithDifferentShape(user, operand_number, new_producer);
}

absl::Status HloInstruction::ReplaceUseWithDifferentShape(
    HloInstruction* user, int operand_number, HloInstruction* new_producer) {
  VLOG(3) << "Replacing operand " << operand_number << " of " << name()
          << " in " << user->name() << " with " << new_producer->name();

  if (absl::c_count(user->operands_, this) == 1) {
    RemoveUser(user);
  }

  TF_RET_CHECK(user->operand(operand_number) == this)
      << "Expected operand " << operand_number << " of " << user->ToString()
      << " to be equal to " << ToString();
  user->operands_[operand_number] = new_producer;
  new_producer->AddUser(user);
  return absl::OkStatus();
}

absl::Status HloInstruction::ReplaceOperandWith(int64_t operand_num,
                                                HloInstruction* new_operand) {
  auto old_operand = operand(operand_num);
  TF_RET_CHECK(
      ShapeUtil::Compatible(old_operand->shape(), new_operand->shape()))
      << old_operand->shape() << " is not compatible with "
      << new_operand->shape();
  return ReplaceOperandWithDifferentShape(operand_num, new_operand);
}

absl::Status HloInstruction::ReplaceOperandWithDifferentShape(
    int64_t operand_num, HloInstruction* new_operand) {
  TF_RET_CHECK(operand_num >= 0);
  TF_RET_CHECK(operand_num < operand_count());
  HloInstruction* old_operand = mutable_operand(operand_num);
  if (old_operand == new_operand) {
    return absl::OkStatus();
  }

  operands_[operand_num] = new_operand;

  VLOG(3) << "Replacing operand " << operand_num << " of " << name() << " with "
          << new_operand->name() << ", was " << old_operand->name();

  if (!absl::c_linear_search(operands_, old_operand)) {
    old_operand->RemoveUser(this);
  }
  new_operand->AddUser(this);
  return absl::OkStatus();
}

absl::Status HloInstruction::ReplaceAllUsesWith(HloInstruction* new_producer,
                                                std::string_view trigger) {
  auto print_options = HloPrintOptions::ShortParsable()
                           .set_print_operand_shape(true)
                           .set_print_extra_attributes(false);
  TF_RET_CHECK(ShapeUtil::Compatible(shape(), new_producer->shape()))
      << "The shape doesn't match when replacing '" << ToString(print_options)
      << "' with '" << new_producer->ToString(print_options) << "'. " << shape()
      << " is not compatible with " << new_producer->shape() << "\n '"
      << trigger << "' triggered this wrong replacement.";
  return ReplaceAllUsesWithDifferentShape(new_producer);
}

absl::Status HloInstruction::ReplaceAllUsesWithDifferentShape(
    HloInstruction* new_producer) {
  bool new_producer_is_user = false;
  // Make a copy since users span might get mutated during the loop
  std::vector<HloInstruction*> users_vector(users().begin(), users().end());
  for (HloInstruction* user : users_vector) {
    if (user == new_producer) {
      // It's possible that new_producer is a user of this instruction as might
      // be the case when replacing an instruction with a kCopy of itself. In
      // this case, don't do the replacement to avoid creating a cycle in the
      // graph. new_producer remains the only user of this instruction.
      new_producer_is_user = true;
    } else {
      std::replace(user->operands_.begin(), user->operands_.end(), this,
                   new_producer);
      new_producer->AddUser(user);
      // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: HloOpcode::kFusion, HloFusionInstruction
      // clang-format on
      // if (user->opcode() == HloOpcode::kFusion) {
      //   TF_RETURN_IF_ERROR(
      //       Cast<HloFusionInstruction>(user)->DeduplicateFusionOperands());
      // }
    }
  }
  users_.Clear();
  if (new_producer_is_user) {
    AddUser(new_producer);
  }
  if (parent_ && parent_->root_instruction() == this) {
    parent_->set_root_instruction(new_producer,
                                  /*accept_different_shape=*/true);
  }

  return absl::OkStatus();
}

HloComputation* HloInstruction::to_apply() const {
  if (has_to_apply()) {
    CHECK_EQ(called_computations().size(), 1)
        << "Expected a to_apply computation for " << opcode();
    return called_computations()[0];
  }
  LOG(FATAL) << "Invalid opcode for to_apply(): " << opcode();
}

void HloInstruction::set_to_apply(HloComputation* computation) {
  if (has_to_apply()) {
    CHECK_EQ(called_computations().size(), 1)
        << "Expected a to_apply computation for " << opcode();
    rare_->called_computations[0] = computation;
    return;
  }
  LOG(FATAL) << "Invalid opcode for to_apply(): " << opcode();
}

bool HloInstruction::has_to_apply() const {
  switch (opcode_) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kScatter:
      return true;
    default:
      return false;
  }
}

int64_t HloInstruction::operand_index(const HloInstruction* target) const {
  for (int64_t i = 0; i < operand_count(); ++i) {
    if (target == operand(i)) {
      return i;
    }
  }
  LOG(FATAL) << "target was not an operand: " << target->ToString();
}

std::vector<int64_t> HloInstruction::operand_indices(
    const HloInstruction* target) const {
  std::vector<int64_t> indices;
  for (int64_t i = 0; i < operand_count(); ++i) {
    if (target == operand(i)) {
      indices.push_back(i);
    }
  }
  if (indices.empty()) {
    LOG(FATAL) << "target was not an operand: " << target->ToString();
  }
  return indices;
}

std::string_view PrintName(std::string_view name, bool print_ids) {
  if (print_ids) {
    return name;
  } else {
    auto dot_position = name.find_first_of('.');
    return name.substr(0, dot_position);
  }
}

namespace {

void PrintNameInternal(Printer* printer, std::string_view name,
                       const HloPrintOptions& options) {
  if (options.print_percent()) {
    printer->Append("%");
  }
  printer->Append(PrintName(name, options.print_ids()));
}

}  //  namespace

void HloInstruction::PrintWithCanonicalNameMap(
    Printer* printer, const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  // Logic to print the instruction name (e.g. "%foo = ").
  if (options.canonicalize_instruction_names()) {
    if (options.is_in_nested_computation()) {
      // If we are canonicalizing instruction names and this is a top-level
      // HloInstruction::ToString() call, don't print an instruction name.
      DCHECK(!options.print_percent());  // no need to call PrintNameInternal
      printer->Append(canonical_name_map->LookupOrInsert(unique_id()));
      printer->Append(" = ");
    }
  } else {
    PrintNameInternal(printer, name(), options);
    printer->Append(" = ");
  }

  if (options.print_result_shape()) {
    // Print shape.
    if (options.include_layout_in_shapes()) {
      ShapeUtil::PrintHumanStringWithLayout(printer, shape());
    } else {
      ShapeUtil::PrintHumanString(printer, shape());
    }
    printer->Append(" ");
  }

  // Print opcode, operand(s).
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: HloInstruction::async_wrapped_computation
  // clang-format on
  //
  // if (options.syntax_sugar_async_ops() && HloOpcodeIsAsync(opcode()) &&
  //     (async_wrapped_computation() &&
  //      async_wrapped_computation()->CanExpandIntoSingleInstruction())) {
  //   std::string_view suffix = [&]() {
  //     switch (opcode()) {
  //       case HloOpcode::kAsyncStart:
  //         return "-start";
  //       case HloOpcode::kAsyncUpdate:
  //         return "-update";
  //       default:
  //         CHECK(opcode() == HloOpcode::kAsyncDone)
  //             << "Unexpected async opcode: " << opcode();
  //         return "-done";
  //     }
  //   }();
  //   printer->Append(HloOpcodeString(async_wrapped_opcode()));
  //   printer->Append(suffix);
  // } else {
  //   printer->Append(HloOpcodeString(opcode()));
  // }
  printer->Append("(");
  PrintOperandsWithCanonicalNameMap(printer, options, canonical_name_map);
  printer->Append(")");

  // Print additional attributes. If an instruction contains a subcomputation,
  // the subcomputation is also printed here.
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // AttributePrinter attr_printer([printer]() {
  //   printer->Append(", ");
  //   return printer;
  // });
  // TODO(chokobole): Uncomment this. Dependency: PrintExtraAttributes
  // PrintExtraAttributes(attr_printer, options);

  if (options.print_original_value() && original_value_) {
    printer->Append(", origin={");
    printer->Append(OriginalValueToString(*original_value()));
    printer->Append("}");
  }

  if (options.print_metadata() &&
      (!metadata_->op_type().empty() || !metadata_->op_name().empty() ||
       !metadata_->source_file().empty() ||
       !metadata_->scheduling_name().empty())) {
    printer->Append(", metadata={");
    printer->Append(
        OpMetadataToString(*metadata_, options.print_metadata_only_op_name()));
    printer->Append("}");
  }
  if (options.print_backend_config() && !backend_config_.empty()) {
    std::string_view config = backend_config_.GetRawString();
    // TODO(chokobole): Uncomment this. Dependency: SortJson
    // std::string sorted_config;
    // if (options.sort_backend_config()) {
    //   // Use `value_or` below, because the backend config string isn't
    //   // guaranteed to be a JSON string.
    //   sorted_config = SortJson(config).value_or(std::string(config));
    //   config = sorted_config;
    // }
    printer->Append(", backend_config=");
    // In the common case that the backend-config is valid-ish JSON, the parser
    // doesn't need it delimited by quotes, so we can print it without
    // CEsape'ing.  This is much easier to read.
    // TODO(chokobole): Uncomment this. Dependency: LexesAsJsonDict
    // if (LexesAsJsonDict(config)) {
    //   printer->Append(config);
    // } else {
    printer->Append("\"");
    printer->Append(absl::CEscape(config));
    printer->Append("\"");
    // TODO(chokobole): Uncomment this. Dependency: LexesAsJsonDict
    // }
  }
}

void HloInstruction::PrintOperandsWithCanonicalNameMap(
    Printer* printer, const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  if (operands_.empty()) return;
  absl::Span<HloInstruction* const> slice(operands_);
  constexpr int64_t kMaxOperandsToShowIfCompact = 4;
  if (options.compact_operands() &&
      slice.size() > kMaxOperandsToShowIfCompact) {
    slice.remove_suffix(slice.size() - kMaxOperandsToShowIfCompact);
  }
  auto print_one = [&](const HloInstruction* operand) {
    // If operand is already been deleted, put `null` to the string output.
    if (operand == nullptr) {
      printer->Append("null ");
      return;
    }
    bool add_space = false;
    if (options.print_operand_shape()) {
      if (options.include_layout_in_shapes()) {
        ShapeUtil::PrintHumanStringWithLayout(printer, operand->shape());
      } else {
        ShapeUtil::PrintHumanString(printer, operand->shape());
      }
      add_space = true;
    }
    if (options.canonicalize_instruction_names()) {
      if (options.is_in_nested_computation()) {
        // In a top-level HloInstruction::ToString() call, the operand name is
        // not part of the canonical string.
        DCHECK(!options.print_percent());  // no need to call PrintNameInternal
        if (add_space) printer->Append(" ");
        printer->Append(
            canonical_name_map->LookupOrInsert(operand->unique_id()));
      }
    } else if (options.print_operand_names()) {
      if (add_space) printer->Append(" ");
      PrintNameInternal(printer, operand->name(), options);
    }
  };
  print_one(slice[0]);
  for (int64_t i = 1; i < slice.size(); ++i) {
    if (options.print_operand_index_annotation_interval() != 0 &&
        i % options.print_operand_index_annotation_interval() == 0) {
      printer->Append(absl::StrFormat(", /*index=%lld*/", i));
    } else {
      printer->Append(", ");
    }
    print_one(slice[i]);
  }
  const int64_t remaining = operands_.size() - slice.size();
  if (remaining > 0) {
    printer->Append(", ...(+");
    printer->Append(remaining);
    printer->Append(")");
  }
}

void HloInstruction::Print(Printer* printer,
                           const HloPrintOptions& options) const {
  CanonicalNameMap new_map;
  PrintWithCanonicalNameMap(printer, options, &new_map);
}

std::string HloInstruction::ToString(const HloPrintOptions& options) const {
  StringPrinter printer;
  Print(&printer, options);
  return std::move(printer).ToString();
}

std::string HloInstruction::ToString() const {
  return ToString(HloPrintOptions::Default());
}

bool HloInstruction::IsOpElementwise(HloOpcode opcode) {
  switch (opcode) {
    // Unary elementwise operations.
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kNegate:
      return true;

    // Binary elementwise operations, the same as in IsElementwiseBinary().
    case HloOpcode::kAdd:
    case HloOpcode::kCompare:
    case HloOpcode::kDivide:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kSubtract:
      return true;

    // Ternary elementwise operations.
    case HloOpcode::kSelect:
      return true;

    default:
      return false;
  }
}

bool HloInstruction::IsElementwiseImpl(
    const std::optional<int64_t>& operand_idx) const {
  if (opcode_ == HloOpcode::kDynamicUpdateSlice) {
    return operand_idx.has_value() && operand_idx.value() == 0;
  }
  if (opcode_ == HloOpcode::kBitcastConvert &&
      primitive_util::BitWidth(shape_.element_type()) !=
          primitive_util::BitWidth(operands_[0]->shape().element_type())) {
    return false;
  }
  return IsOpElementwise(opcode_);
}

std::string FrontendAttributesToString(
    const FrontendAttributes& frontend_attributes) {
  std::vector<std::pair<std::string, std::string>> sorted_attributes(
      frontend_attributes.map().begin(), frontend_attributes.map().end());
  absl::c_sort(sorted_attributes);
  const auto formatter = [](std::string* out,
                            const std::pair<std::string, std::string>& item) {
    // TODO(chokobole): Uncomment this. Dependency: LexesAsJsonDict
    // if (LexesAsJsonDict(item.second)) {
    //   absl::StrAppend(out, item.first, "=", item.second);
    // } else {
    absl::StrAppend(out, item.first, "=\"", item.second, "\"");
    // TODO(chokobole): Uncomment this. Dependency: LexesAsJsonDict
    // }
  };
  return absl::StrFormat("{%s}",
                         absl::StrJoin(sorted_attributes, ",", formatter));
}

HloInstruction::HloInstruction(HloOpcode opcode, const Shape& shape)
    : unique_id_(-1),
      index_in_parent_(~0u),
      opcode_(opcode),
      is_default_config_(false),
      cleaned_up_(false),
      marked_as_dead_(false),
      is_root_(false),
      shape_(shape),
      name_(HloOpcodeString(opcode)) {
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape_));
}

absl::InlinedVector<int64_t, 4> HloInstruction::OperandIndices(
    const HloInstruction* operand) const {
  absl::InlinedVector<int64_t, 4> result;
  for (int64_t i = 0; i < operand_count(); ++i) {
    if (this->operand(i) == operand) {
      result.push_back(i);
    }
  }
  return result;
}

bool HloInstruction::IsElementwise() const {
  return IsElementwiseImpl(std::nullopt);
}

bool HloInstruction::IsElementwiseOnOperand(int64_t operand_idx) const {
  return IsElementwiseImpl(operand_idx);
}

HloModule* HloInstruction::GetModule() const {
  if (parent_) {
    return parent_->parent();
  }
  return nullptr;
}

void HloInstruction::UniquifyName(HloModule* module) {
  UniquifyName(&module->instruction_name_uniquer());
}

int64_t HloInstruction::parameter_number() const {
  return Cast<HloParameterInstruction>(this)->parameter_number();
}

std::optional<int64_t> HloInstruction::channel_id() const {
  return Cast<HloChannelInstruction>(this)->channel_id();
}

void HloInstruction::set_channel_id(const std::optional<int64_t>& channel_id) {
  return Cast<HloChannelInstruction>(this)->set_channel_id(channel_id);
}

bool HloInstruction::IsAsynchronous() const {
  return HloOpcodeIsAsync(opcode());
}

HloInstruction* HloInstruction::async_chain_start() const {
  return Cast<HloAsyncInstruction>(this)->async_chain_start();
}

HloInstruction* HloInstruction::async_chain_done() const {
  return Cast<HloAsyncInstruction>(this)->async_chain_done();
}

HloComputation* HloInstruction::async_wrapped_computation() const {
  return Cast<HloAsyncInstruction>(this)->async_wrapped_computation();
}

HloInstruction* HloInstruction::async_wrapped_instruction() const {
  return Cast<HloAsyncInstruction>(this)->async_wrapped_instruction();
}

HloOpcode HloInstruction::async_wrapped_opcode() const {
  return Cast<HloAsyncInstruction>(this)->async_wrapped_opcode();
}

std::string_view HloInstruction::async_execution_thread() const {
  return Cast<HloAsyncInstruction>(this)->async_execution_thread();
}

void HloInstruction::set_async_execution_thread(
    std::string_view async_execution_thread) {
  Cast<HloAsyncInstruction>(this)->set_async_execution_thread(
      async_execution_thread);
}

}  // namespace zkx
