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

#include "zkx/hlo/ir/hlo_module.h"

#include "absl/strings/escaping.h"

#include "xla/tsl/platform/fingerprint.h"
#include "xla/tsl/platform/statusor.h"

namespace zkx {

absl::Status HloModule::set_schedule(HloSchedule schedule) {
  TF_RET_CHECK(schedule.module() == this);
  TF_RETURN_IF_ERROR(schedule.Verify());
  schedule_ = std::move(schedule);
  return absl::OkStatus();
}

void HloModule::ReplaceEntryComputation(HloComputation* entry_computation) {
  entry_computation_ = entry_computation;
  mutable_config().SetDefaultComputationLayout(
      entry_computation_->ComputeProgramShape());
  input_output_alias_config_ = HloInputOutputAliasConfig(
      entry_computation_->root_instruction()->shape());
  buffer_donor_config_ = HloBufferDonorConfig();
}

HloModule::StackFrame HloModule::get_stack_frame(int id) const {
  HloModule::StackFrame stack_frame;
  if (!stack_frame_index_.has_value() || id < 1 ||
      id > stack_frame_index_->stack_frames().size()) {
    return stack_frame;
  }

  auto& frame = stack_frame_index_->stack_frames(id - 1);
  auto& file_location =
      stack_frame_index_->file_locations(frame.file_location_id() - 1);

  stack_frame.file_name =
      stack_frame_index_->file_names(file_location.file_name_id() - 1);
  stack_frame.function_name =
      stack_frame_index_->function_names(file_location.function_name_id() - 1);
  stack_frame.line = file_location.line();
  stack_frame.column = file_location.column();
  stack_frame.parent_frame_id = frame.parent_frame_id();

  return stack_frame;
}

HloComputation* HloModule::AddComputationInternal(
    std::unique_ptr<HloComputation> computation, bool is_entry,
    bool uniquify_identifiers, bool preserve_entry_layouts) {
  if (is_entry) {
    CHECK_EQ(nullptr, entry_computation_);
    entry_computation_ = computation.get();

    if (preserve_entry_layouts) {
      mutable_config().SetComputationLayoutIfExists(
          entry_computation_->ComputeProgramShape());
    } else if (!config().has_entry_computation_layout()) {
      // If the module configuration has no entry layout computation set, create
      // a default one based on the program shape.
      mutable_config().SetDefaultComputationLayout(
          entry_computation_->ComputeProgramShape());
    }
    input_output_alias_config_ = HloInputOutputAliasConfig(
        entry_computation_->root_instruction()->shape());
    buffer_donor_config_ = HloBufferDonorConfig();
  }

  if (uniquify_identifiers) {
    computation->UniquifyName(&computation_name_uniquer_);
    for (auto* instruction : computation->instructions()) {
      instruction->UniquifyName(&instruction_name_uniquer_);
    }

    // Pick unique IDs for each instruction.
    for (auto* instruction : computation->instructions()) {
      instruction->SetUniqueId(NewUniqueInstructionId());
    }
    // Set unique id to this computation.
    CHECK_NE(computation->root_instruction()->unique_id(), -1)
        << "Root has no valid id: " << computation->ToString();
    computation->SetUniqueId(computation->root_instruction()->unique_id());
  } else {
    // Don't uniquify the names of the computation or instruction, but we must
    // run the names through the uniquifiers to prevent future name collisions
    // for computations and instructions created later. Also, set the
    // next_unique_id_ to the one greater than the max unique id of any
    // instruction (or the computation) to avoid ID collisions.
    computation_name_uniquer_.GetUniqueName(computation->name());
    for (auto* instruction : computation->instructions()) {
      instruction_name_uniquer_.GetUniqueName(instruction->name());
      next_unique_id_ = std::max(next_unique_id_, instruction->unique_id() + 1);
    }
    if (next_unique_id_ < computation->unique_id() + 1) {
      next_unique_id_ = computation->unique_id() + 1;
    }
  }

  computation->set_parent(this);
  computations_.push_back(std::move(computation));
  return computations_.back().get();
}

HloComputation* HloModule::AddEntryComputation(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/true,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/false);
}

HloComputation* HloModule::AddEntryComputationWithLayouts(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/true,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/true);
}

absl::Status HloModule::RemoveEmbeddedComputation(HloComputation* to_remove) {
  if (has_schedule()) {
    schedule_->remove_computation(to_remove);
  }

  auto it = absl::c_find_if(
      computations_, [&to_remove](const std::unique_ptr<HloComputation>& comp) {
        return comp.get() == to_remove;
      });
  TF_RET_CHECK(it != computations_.end());
  TF_RET_CHECK(it->get() == to_remove);
  computations_.erase(it);
  return absl::OkStatus();
}

HloComputation* HloModule::AddEmbeddedComputation(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/false,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/false);
}

void HloModule::MoveComputationsFrom(HloModule* module,
                                     bool make_names_unique) {
  for (size_t i = 0; i < module->computation_count(); ++i) {
    for (auto* instruction : module->computations_[i]->instructions()) {
      instruction->ClearUniqueIdInternal();
    }
    module->computations_[i]->ClearUniqueIdInternal();
    auto computation_raw_ptr = module->computations_[i].get();
    if (computation_raw_ptr->IsEntryComputation()) {
      this->entry_computation_ = nullptr;
    }
    this->AddComputationInternal(
        std::move(module->computations_[i]),
        /*is_entry=*/computation_raw_ptr->IsEntryComputation(),
        /*uniquify_identifiers=*/false,
        /*preserve_entry_layouts=*/false);
    if (make_names_unique) {
      computation_raw_ptr->UniquifyName(&computation_name_uniquer_);
      for (auto* instruction : computation_raw_ptr->instructions()) {
        instruction->UniquifyName(&instruction_name_uniquer_);
      }
    }
    // Pick unique IDs for each instruction.
    for (auto* instruction : computation_raw_ptr->instructions()) {
      instruction->SetUniqueId(NewUniqueInstructionId());
    }
    // Set unique id to this computation_raw_ptr.
    CHECK_NE(computation_raw_ptr->root_instruction()->unique_id(), -1)
        << "Root has no valid id: " << computation_raw_ptr->ToString();
    computation_raw_ptr->SetUniqueId(
        computation_raw_ptr->root_instruction()->unique_id());
  }
  // Since the computations no longer belong to the old module, clear the list.
  module->computations_.clear();
}

void HloModule::Print(Printer* printer, const HloPrintOptions& options) const {
  printer->Append("HloModule ");
  if (options.print_ids()) {
    // When print_ids() is false, exclude module's name because it includes and
    // leads to non-deterministic fingerprint.
    printer->Append(name());
  }
  if (has_schedule()) {
    TF_CHECK_OK(schedule().Verify());
    printer->Append(", is_scheduled=true");
  }
  std::string serialized_aliasing = input_output_alias_config().ToShortString();
  if (!serialized_aliasing.empty()) {
    printer->Append(", input_output_alias={ ");
    printer->Append(std::move(serialized_aliasing));
    printer->Append(" }");
  }
  std::string serialized_buffer_donor = buffer_donor_config().ToShortString();
  if (!serialized_buffer_donor.empty()) {
    printer->Append(", buffer_donor={ ");
    printer->Append(std::move(serialized_buffer_donor));
    printer->Append(" }");
  }

  const HloModuleConfig& config = this->config();
  if (config.alias_passthrough_params()) {
    printer->Append(", alias_passthrough_params=true");
  }
  if (config.has_entry_computation_layout()) {
    printer->Append(", entry_computation_layout={");
    entry_computation_layout().Print(printer);
    printer->Append("}");
  }
  if (config.allow_spmd_sharding_propagation_to_parameters().size() != 1 ||
      config.allow_spmd_sharding_propagation_to_parameters().back()) {
    printer->Append(", allow_spmd_sharding_propagation_to_parameters={");
    AppendJoin(printer, config.allow_spmd_sharding_propagation_to_parameters(),
               ",", [](Printer* printer, bool i) {
                 printer->Append(i ? "true" : "false");
               });
    printer->Append("}");
  }
  if (config.allow_spmd_sharding_propagation_to_output().size() != 1 ||
      config.allow_spmd_sharding_propagation_to_output().back()) {
    printer->Append(", allow_spmd_sharding_propagation_to_output={");
    AppendJoin(printer, config.allow_spmd_sharding_propagation_to_output(), ",",
               [](Printer* printer, bool i) {
                 printer->Append(i ? "true" : "false");
               });
    printer->Append("}");
  }
  if (config.replica_count() != 1) {
    printer->Append(", replica_count=");
    printer->Append(config.replica_count());
  }
  if (config.num_partitions() != 1) {
    printer->Append(", num_partitions=");
    printer->Append(config.num_partitions());
  }
  if (!frontend_attributes_.map().empty()) {
    AppendCat(printer, ", frontend_attributes=",
              FrontendAttributesToString(frontend_attributes_));
  }
  printer->Append("\n\n");
  const auto& computations = options.canonicalize_computations()
                                 ? MakeComputationSorted()
                                 : MakeComputationPostOrder();
  for (const HloComputation* computation : computations) {
    // Don't print async computations when the syntax sugar is enabled since
    // that is redundant information.
    if (options.syntax_sugar_async_ops() && computation->IsAsyncComputation() &&
        computation->CanExpandIntoSingleInstruction()) {
      continue;
    }
    if (computation == entry_computation()) {
      printer->Append("ENTRY ");
    }
    if (has_schedule() && schedule().is_computation_scheduled(computation)) {
      computation->Print(printer, options,
                         schedule().sequence(computation).instructions());
    } else {
      computation->Print(printer, options);
    }
    printer->Append("\n\n");
  }
}

std::string HloModule::ToString(const HloPrintOptions& options) const {
  StringPrinter printer;
  Print(&printer, options);
  return std::move(printer).ToString();
}

absl::Cord HloModule::ToCord(const HloPrintOptions& options) const {
  CordPrinter printer;
  Print(&printer, options);
  return std::move(printer).ToCord();
}

absl::Status HloModule::CheckUniqueNamesAndIdsForComputationsAndInstructions()
    const {
  absl::flat_hash_set<std::string_view> computation_names;
  absl::flat_hash_set<int> computation_ids;
  absl::flat_hash_set<std::string_view> instruction_names;
  absl::flat_hash_set<int> instruction_ids;

  for (const HloComputation* computation : computations()) {
    TF_RET_CHECK(!ContainsKey(computation_names, computation->name()))
        << "Computation name is not unique: " << computation->name();
    computation_names.insert(computation->name());

    TF_RET_CHECK(!ContainsKey(computation_ids, computation->unique_id()))
        << "Computation id is not unique: " << computation->unique_id();
    computation_ids.insert(computation->unique_id());

    for (const HloInstruction* instruction : computation->instructions()) {
      TF_RET_CHECK(!ContainsKey(instruction_names, instruction->name()))
          << "Instruction name is not unique: " << instruction->name();
      instruction_names.insert(instruction->name());

      TF_RET_CHECK(!ContainsKey(instruction_ids, instruction->unique_id()))
          << "Instruction id is not unique: " << instruction->unique_id();
      instruction_ids.insert(instruction->unique_id());
    }
  }
  return absl::OkStatus();
}

int64_t HloModule::instruction_count() const {
  int64_t n = 0;
  for (const auto& computation : computations_) {
    n += computation->instruction_count();
  }
  return n;
}

std::vector<HloComputation*> HloModule::MakeComputationPostOrder(
    const absl::flat_hash_set<std::string_view>& execution_threads,
    const absl::flat_hash_set<HloComputation*>& allow_list) const {
  std::vector<HloComputation*> post_order =
      this->MakeComputationPostOrder(execution_threads);

  post_order.erase(std::remove_if(post_order.begin(), post_order.end(),
                                  [&allow_list](HloComputation* computation) {
                                    return !allow_list.contains(computation);
                                  }),
                   post_order.end());

  return post_order;
}

std::vector<HloComputation*> HloModule::MakeComputationPostOrder(
    const absl::flat_hash_set<std::string_view>& execution_threads) const {
  if (computations_.empty()) {
    return {};
  }
  // First determine all root computations by building a set of non-root
  // computations (computations which are called by an instruction in the
  // module).
  absl::flat_hash_set<HloComputation*> nonroot_computations;
  nonroot_computations.reserve(computations_.size() - 1);
  for (auto& computation : computations_) {
    for (const HloInstructionInfo& inst :
         computation->instructions_with_info()) {
      if (HloInstruction::MightHaveCalledComputations(inst.opcode())) {
        for (HloComputation* called_computation : inst->called_computations()) {
          nonroot_computations.insert(called_computation);
        }
      }
    }
  }

  // Keep track of computations which have already been added to the post
  // order. This prevents duplication as an embedded computation may be called
  // from two different root computations.
  absl::flat_hash_set<HloComputation*> added_computations;
  std::vector<HloComputation*> post_order;
  added_computations.reserve(computations_.size());
  post_order.reserve(computations_.size());
  for (auto& computation : computations_) {
    if (nonroot_computations.contains(computation.get())) {
      continue;
    }
    for (HloComputation* embedded_computation :
         computation->MakeEmbeddedComputationsList()) {
      if (added_computations.insert(embedded_computation).second) {
        post_order.push_back(embedded_computation);
      }
    }
    // Root computations should only be encountered once.
    CHECK(!added_computations.contains(computation.get()));
    post_order.push_back(computation.get());
    added_computations.insert(computation.get());
  }
  if (post_order.size() != computations_.size()) {
    for (HloComputation* computation : post_order) {
      LOG(ERROR) << "Post Order: " << computation->name() << " ("
                 << computation->parent()->name() << ")";
    }
    for (auto& computation : computations_) {
      LOG(ERROR) << "Computations: " << computation->name() << " ("
                 << computation->parent()->name() << ")";
    }
    LOG(FATAL) << "Mismatch computation count: post_order=" << post_order.size()
               << " computation_count=" << computations_.size();
  }
  if (!execution_threads.empty()) {
    post_order.erase(std::remove_if(post_order.begin(), post_order.end(),
                                    [&](HloComputation* computation) {
                                      return !execution_threads.contains(
                                          computation->execution_thread());
                                    }),
                     post_order.end());
  }
  return post_order;
}

namespace {

class FingerprintMap {
 public:
  void Reserve(int capacity) { fingerprint_map_.reserve(capacity); }

  uint64_t GetFingerprint(const HloComputation* computation) {
    auto result = fingerprint_map_.try_emplace(computation, 0);
    if (result.second) {
      result.first->second =
          tsl::Fingerprint64(computation->ToString(print_options_));
    }
    return result.first->second;
  }

 private:
  HloPrintOptions print_options_ = HloPrintOptions::ModuleFingerprint();
  absl::flat_hash_map<const HloComputation*, uint64_t> fingerprint_map_;
};

void SortComputationsByContent(std::vector<HloComputation*>* computations) {
  FingerprintMap fingerprint_map;
  fingerprint_map.Reserve(computations->size());
  auto cmp = [&fingerprint_map](const HloComputation* a,
                                const HloComputation* b) {
    if (a->instruction_count() != b->instruction_count()) {
      return a->instruction_count() < b->instruction_count();
    }
    // Avoid computing fingerprints of (potentially) giant computation strings
    // just to compare when a == b
    if (a == b) return false;

    return fingerprint_map.GetFingerprint(a) <
           fingerprint_map.GetFingerprint(b);
  };
  absl::c_sort(*computations, cmp);
}

}  // anonymous namespace

std::vector<HloComputation*> HloModule::MakeComputationSorted(
    const absl::flat_hash_set<std::string_view>& execution_threads) const {
  std::vector<HloComputation*> result =
      MakeComputationPostOrder(execution_threads);
  if (config().content_aware_computation_sorting()) {
    SortComputationsByContent(&result);
  }
  return result;
}

std::vector<HloComputation*> HloModule::MakeNonfusionComputations(
    const absl::flat_hash_set<std::string_view>& execution_threads) const {
  std::vector<HloComputation*> result =
      MakeComputationPostOrder(execution_threads);
  result.erase(std::remove_if(
                   result.begin(), result.end(),
                   [](HloComputation* c) { return c->IsFusionComputation(); }),
               result.end());
  return result;
}

std::vector<HloComputation*> HloModule::MakeNonfusionComputationsSorted(
    const absl::flat_hash_set<std::string_view>& execution_threads) const {
  auto result = MakeNonfusionComputations(execution_threads);
  if (config().content_aware_computation_sorting()) {
    SortComputationsByContent(&result);
  }
  return result;
}

absl::Status HloModule::RemoveUnusedComputations() {
  absl::flat_hash_set<HloComputation*> to_remove(computations().begin(),
                                                 computations().end());
  std::stack<HloComputation*> agenda;
  agenda.push(entry_computation_);
  to_remove.erase(entry_computation_);
  while (!agenda.empty()) {
    HloComputation* computation = agenda.top();
    agenda.pop();
    for (HloInstruction* instruction : computation->instructions()) {
      for (HloComputation* called_computation :
           instruction->called_computations()) {
        if (to_remove.erase(called_computation) > 0) {
          agenda.push(called_computation);
        }
      }
    }
  }
  for (auto computation : to_remove) {
    TF_RETURN_IF_ERROR(RemoveEmbeddedComputation(computation));
  }
  return absl::OkStatus();
}

uint64_t HloModule::RandomNew64() const {
  absl::MutexLock l(&rng_mutex_);
  return rng_();
}

HloComputation* HloModule::GetComputationWithName(std::string_view name) {
  auto computations_in_module = computations();
  auto it = absl::c_find_if(
      computations_in_module,
      [&](HloComputation* computation) { return computation->name() == name; });
  return it == computations_in_module.end() ? nullptr : *it;
}

std::string HloModule::GetFingerprint128(const HloPrintOptions& options) const {
  const tsl::Fprint128 fingerprint = tsl::Fingerprint128(ToString(options));
  std::string_view fp_bytes(reinterpret_cast<const char*>(&fingerprint),
                            sizeof(tsl::Fprint128));
  return absl::BytesToHexString(fp_bytes);
}

// static
std::atomic<int> HloModule::next_unique_module_id_(0);

}  // namespace zkx
