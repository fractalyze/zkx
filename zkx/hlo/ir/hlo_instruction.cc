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
#include "absl/log/log.h"

#include "zkx/shape_util.h"

namespace zkx {

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

  // TODO(chokobole): Uncomment this. Dependency: metadata_
  // if (options.print_metadata() &&
  //     (!metadata_->op_type().empty() || !metadata_->op_name().empty() ||
  //      !metadata_->source_file().empty() ||
  //      !metadata_->scheduling_name().empty())) {
  //   printer->Append(", metadata={");
  //   printer->Append(OpMetadataToString(
  //       *metadata_, options.print_metadata_only_op_name()));
  //   printer->Append("}");
  // }
  // if (options.print_backend_config() && !backend_config_.empty()) {
  //   std::string_view config = backend_config_.GetRawString();
  //   std::string sorted_config;
  //   if (options.sort_backend_config()) {
  //     // Use `value_or` below, because the backend config string isn't
  //     // guaranteed to be a JSON string.
  //     sorted_config = SortJson(config).value_or(std::string(config));
  //     config = sorted_config;
  //   }
  //   printer->Append(", backend_config=");
  //   // In the common case that the backend-config is valid-ish JSON, the
  //   parser
  //   // doesn't need it delimited by quotes, so we can print it without
  //   // CEsape'ing.  This is much easier to read.
  //   if (LexesAsJsonDict(config)) {
  //     printer->Append(config);
  //   } else {
  //     printer->Append("\"");
  //     printer->Append(CEscape(config));
  //     printer->Append("\"");
  //   }
  // }
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

}  // namespace zkx
