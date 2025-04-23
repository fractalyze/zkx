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

#include "zkx/hlo/analysis/hlo_dataflow_analysis.h"

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"

namespace zkx {

HloDataflowAnalysis::HloDataflowAnalysis(
    const HloModule& module, bool ssa_form, bool bitcast_defines_value,
    const CanShareBuffer& can_share_buffer, const ForwardsValue& forwards_value,
    absl::flat_hash_set<std::string_view> execution_threads)
    : module_(module),
      execution_threads_(std::move(execution_threads)),
      // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: ssa_form_, bitcast_defines_value_
      // clang-format on
      // ssa_form_(ssa_form),
      // bitcast_defines_value_(bitcast_defines_value),
      call_graph_(CallGraph::Build(&module)),
      can_share_buffer_(can_share_buffer),
      forwards_value_(forwards_value) {}

bool HloDataflowAnalysis::ValueIsDefinedAt(const HloInstruction* instruction,
                                           const ShapeIndex& index) const {
  const HloValueSet& value_set = GetValueSet(instruction, index);
  if (value_set.values().size() != 1) {
    return false;
  }
  return value_set.GetUniqueValue().defining_instruction() == instruction;
}

const HloValue& HloDataflowAnalysis::GetValueDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  CHECK(ValueIsDefinedAt(instruction, index)) << instruction->ToString();
  return GetUniqueValueAt(instruction, index);
}

HloValue& HloDataflowAnalysis::GetValueDefinedAt(
    const HloInstruction* instruction, const ShapeIndex& index) {
  CHECK(ValueIsDefinedAt(instruction, index));
  return GetUniqueValueAt(instruction, index);
}

HloValue* HloDataflowAnalysis::NewHloValue(HloInstruction* instruction,
                                           const ShapeIndex& index,
                                           bool is_phi) {
  const int64_t value_id = next_value_id_++;
  auto result =
      values_.insert({value_id, std::make_unique<HloValue>(
                                    value_id, instruction, index, is_phi)});
  CHECK(result.second);

  VLOG(4) << "NewHloValue = " << result.first->second->ToShortString();

  return result.first->second.get();
}

std::string HloDataflowAnalysis::ToString() const {
  std::string out =
      absl::StrCat("HloDataflowAnalysis, module ", module_.name(), "\n");
  absl::StrAppend(&out, "  Instruction value sets:\n");
  for (const HloComputation* computation : module_.computations()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    for (const HloInstruction* instruction : computation->instructions()) {
      absl::StrAppend(&out, "Instruction: \n  ", instruction->name(), ":\n");
      if (instruction->shape().IsTuple()) {
        GetInstructionValueSet(instruction)
            .ForEachElement([this, &instruction, &out](
                                const ShapeIndex& index,
                                const HloValueSet& value_set) {
              absl::StrAppend(&out, "      tuple index ", index.ToString(),
                              ":\n");
              for (const HloValue* value : value_set.values()) {
                absl::StrAppend(
                    &out, "        ", value->ToShortString(),
                    ValueIsDefinedAt(instruction, index) ? " (def)" : "", "\n");
              }
            });
      } else {
        const HloValueSet& top_level_value_set =
            GetValueSet(instruction, /*index=*/{});
        for (const HloValue* value : top_level_value_set.values()) {
          absl::StrAppend(&out, "      ", value->ToShortString(),
                          ValueIsDefinedAt(instruction) ? " (def)" : "", "\n");
        }
      }
    }
  }
  absl::StrAppend(&out, "  HloValues:\n");
  for (const HloValue* value : values()) {
    absl::StrAppend(&out, value->ToString(/*indent=*/4));
  }
  return out;
}

const HloValue& HloDataflowAnalysis::GetValue(HloValue::Id value_id) const {
  DCHECK(values_.contains(value_id)) << "Value not found: " << value_id;
  return *values_.find(value_id)->second;
}

HloValue& HloDataflowAnalysis::GetValue(HloValue::Id value_id) {
  DCHECK(values_.contains(value_id)) << "Value not found: " << value_id;
  return *values_.find(value_id)->second;
}

HloValueSet HloDataflowAnalysis::GetFlattenedValueSet(
    const HloInstruction* instruction) const {
  HloValueSet value_set;

  const InstructionValueSet& value_set_tree =
      GetInstructionValueSet(instruction);

  std::vector<const HloValueSet*> all_sets;
  for (auto& pair : value_set_tree) {
    const HloValueSet& value_set = pair.second;
    all_sets.push_back(&value_set);
  }
  value_set.AssignUnionOf(all_sets);

  return value_set;
}

const HloValueSet& HloDataflowAnalysis::GetValueSet(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  return GetInstructionValueSet(instruction).element(index);
}

HloValueSet& HloDataflowAnalysis::GetValueSet(const HloInstruction* instruction,
                                              const ShapeIndex& index) {
  return *GetInstructionValueSet(instruction).mutable_element(index);
}

const HloValueSet& HloDataflowAnalysis::GetValueSet(
    const HloPosition& position) const {
  return GetValueSet(position.instruction, position.index);
}

HloValueSet& HloDataflowAnalysis::GetValueSet(const HloPosition& position) {
  return GetValueSet(position.instruction, position.index);
}

const InstructionValueSet& HloDataflowAnalysis::GetInstructionValueSet(
    const HloInstruction* instruction) const {
  DCHECK(value_sets_.contains(instruction))
      << "Instruction " << instruction->ToString() << " not found.";
  return *value_sets_.find(instruction)->second;
}

InstructionValueSet& HloDataflowAnalysis::GetInstructionValueSet(
    const HloInstruction* instruction) {
  DCHECK(value_sets_.contains(instruction))
      << "Instruction " << instruction->ToString() << " not found.";
  return *value_sets_.find(instruction)->second;
}

// static
bool HloDataflowAnalysis::IsInPlaceOperation(HloOpcode opcode) {
  return opcode == HloOpcode::kDynamicUpdateSlice ||
         opcode == HloOpcode::kScatter;
}

// static
bool HloDataflowAnalysis::IsAsynchronousOperationStart(HloOpcode opcode) {
  return opcode == HloOpcode::kSend || opcode == HloOpcode::kRecv ||
         opcode == HloOpcode::kCopyStart ||
         opcode == HloOpcode::kAllReduceStart ||
         opcode == HloOpcode::kAllGatherStart ||
         opcode == HloOpcode::kCollectivePermuteStart ||
         opcode == HloOpcode::kAsyncStart;
}

// static
bool HloDataflowAnalysis::IsAsynchronousOperationDone(HloOpcode opcode) {
  return opcode == HloOpcode::kSendDone || opcode == HloOpcode::kRecvDone ||
         opcode == HloOpcode::kCopyDone ||
         opcode == HloOpcode::kAllReduceDone ||
         opcode == HloOpcode::kAllGatherDone ||
         opcode == HloOpcode::kCollectivePermuteDone ||
         opcode == HloOpcode::kAsyncDone;
}

}  // namespace zkx
