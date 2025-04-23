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

// Analysis for determining the possible set of values for all positions
// (instructions and ShapeIndexes) in the HLO module. Analysis is module-scoped
// tracking values across computation boundaries.

#ifndef ZKX_HLO_ANALYSIS_HLO_DATAFLOW_ANALYSIS_H_
#define ZKX_HLO_ANALYSIS_HLO_DATAFLOW_ANALYSIS_H_

#include <stdint.h>

#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/service/call_graph.h"
#include "zkx/service/hlo_value.h"
#include "zkx/service/phi_graph.h"
#include "zkx/shape_util.h"

namespace zkx {

// Identifies one array input of an HloInstruction.
struct HloOperandIndex {
  using MyTuple = std::tuple<int64_t, const ShapeIndex&>;

  template <typename H>
  friend H AbslHashValue(H h, const HloOperandIndex& hlo_operand_index) {
    return H::combine(std::move(h), hlo_operand_index.ToTuple());
  }

  friend bool operator==(const HloOperandIndex& lhs,
                         const HloOperandIndex& rhs) {
    return lhs.ToTuple() == rhs.ToTuple();
  }

  bool operator!=(const HloOperandIndex& other) const {
    return !(*this == other);
  }

  MyTuple ToTuple() const {
    return std::make_tuple(operand_number, std::cref(operand_index));
  }

  // The operand number in which the array value appears.
  int64_t operand_number;

  // The shape index within the operand in which the array value appears.
  ShapeIndex operand_index;
};

// Analysis which identifies all HLO values and their uses in an HLO module.
class HloDataflowAnalysis {
 public:
  // Infrastructure for passing may-alias hints: HLO passes can populate the
  // may-alias table. If an empty optional is returned, default rules are used.
  //
  // Must-alias rules (as defined by GetInPlaceInputOutputPairs) cannot be
  // overriden using backend-specific overrides.
  //
  // The first parameter of the function should be the instruction, the
  // second parameter should be an operand of the instruction. The third
  // parameter should be the output index of the instruction.
  using CanShareBuffer = std::function<std::optional<bool>(
      const HloInstruction* instr, const HloInstruction* operand,
      const ShapeIndex& user_index)>;

  // Infrastructure for overriding whether an instruction defines a new value.
  //
  // The first parameter is the instruction and the second parameter is the
  // output index. If an empty optional is used, default rules are used. If a
  // ForwardedOperand object is returned, the value at the corresponding
  // operand's index is used for the output, overriding all default logic.
  struct ForwardedOperand {
    int64_t operand_number;
    ShapeIndex operand_index;
  };
  using ForwardsValue = std::function<std::optional<ForwardedOperand>(
      const HloInstruction* instr, const ShapeIndex& index)>;

  // Returns true if 'instruction' defines an HLO value at the given shape index
  // of its output.
  bool ValueIsDefinedAt(const HloInstruction* instruction,
                        const ShapeIndex& index = {}) const;

  // Returns the HloValue defined by 'instruction' at the given shape index of
  // its output.
  //
  // Precondition: ValueIsDefinedAt is true for this instruction and index.
  const HloValue& GetValueDefinedAt(const HloInstruction* instruction,
                                    const ShapeIndex& index = {}) const;
  HloValue& GetValueDefinedAt(const HloInstruction* instruction,
                              const ShapeIndex& index = {});

  // Returns the InstructionValueSet for the given instruction.
  const InstructionValueSet& GetInstructionValueSet(
      const HloInstruction* instruction) const;
  InstructionValueSet& GetInstructionValueSet(
      const HloInstruction* instruction);

  // Returns all values that are contained in the output of this instruction in
  // a flattened set.
  HloValueSet GetFlattenedValueSet(const HloInstruction* instruction) const;

  // Returns the HloValueSet for the given instruction at the given index or the
  // given position.
  const HloValueSet& GetValueSet(const HloInstruction* instruction,
                                 const ShapeIndex& index = {}) const;
  const HloValueSet& GetValueSet(const HloPosition& position) const;
  HloValueSet& GetValueSet(const HloPosition& position);
  HloValueSet& GetValueSet(const HloInstruction* instruction,
                           const ShapeIndex& index = {});

  // Returns the unique value in the HloValueSet at the given instruction and
  // shape index. CHECKs if the value set does not contain a exactly one value.
  const HloValue& GetUniqueValueAt(const HloInstruction* instruction,
                                   const ShapeIndex& index = {}) const {
    return GetValueSet(instruction, index).GetUniqueValue();
  }
  HloValue& GetUniqueValueAt(const HloInstruction* instruction,
                             const ShapeIndex& index = {}) {
    return GetValue(GetValueSet(instruction, index).GetUniqueValue().id());
  }

  // Returns the HloValue with the given Id.
  const HloValue& GetValue(HloValue::Id value_id) const;
  HloValue& GetValue(HloValue::Id value_id);

  // Returns the total number of HloValues.
  int64_t value_count() const { return values_.size(); }

  // Returns a vector of all HloValues stably sorted by HloValue::Id.
  const std::vector<HloValue*>& values() const { return values_vector_; }

  // Returns the call graph used for computing the dataflow.
  const CallGraph& call_graph() const { return *call_graph_; }

  std::string ToString() const;

  const HloModule& module() const { return module_; }

  // Returns true if the operation is an in-place operation and its operand 0
  // must alias with the output.
  static bool IsInPlaceOperation(HloOpcode opcode);

  // Returns true if the operation is the start/done of an asynchronous
  // operation, where the buffer used/produced by the op needs to stay alive
  // until the asynchronous operation completes.
  static bool IsAsynchronousOperationStart(HloOpcode opcode);
  static bool IsAsynchronousOperationDone(HloOpcode opcode);

 private:
  HloDataflowAnalysis(const HloModule& module, bool ssa_form,
                      bool bitcast_defines_value,
                      const CanShareBuffer& can_share_buffer,
                      const ForwardsValue& forwards_value,
                      absl::flat_hash_set<std::string_view> execution_threads);

  // Returns a new HloValue defined at the given instruction and shape index.
  HloValue* NewHloValue(HloInstruction* instruction, const ShapeIndex& index,
                        bool is_phi);

  const HloModule& module_;
  const absl::flat_hash_set<std::string_view> execution_threads_;

  std::unique_ptr<CallGraph> call_graph_;

  // The map of all HloValues in the module. We pass around pointers to the
  // mapped HloValues, so the underlying container must keep them valid despite
  // mutations touching other map entries.
  absl::flat_hash_map<HloValue::Id, std::unique_ptr<HloValue>> values_;

  // A map from instruction to InstructionValueSet.
  absl::flat_hash_map<const HloInstruction*,
                      std::unique_ptr<InstructionValueSet>>
      value_sets_;

  // Values marked for deletion during construction. We don't delete them
  // immediately because references to them may remain in ValueSets temporarily
  // during propagation. After construction, these values are deleted.
  std::vector<HloValue::Id> value_ids_to_delete_;

  // A vector containing all HloValues sorted by HloValue::Id.
  std::vector<HloValue*> values_vector_;

  // The Id to use for the next HloValue.
  HloValue::Id next_value_id_ = 0;

  // An explicit graph holding phi values and edges.
  PhiGraph phi_graph_;

  // Backend specific function that decides whether an instruction can share
  // a buffer with its operand.
  CanShareBuffer can_share_buffer_ = nullptr;

  ForwardsValue forwards_value_ = nullptr;
};

}  // namespace zkx

#endif  // ZKX_HLO_ANALYSIS_HLO_DATAFLOW_ANALYSIS_H_
