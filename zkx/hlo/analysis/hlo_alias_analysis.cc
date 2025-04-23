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

#include "zkx/hlo/analysis/hlo_alias_analysis.h"

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"

#include "zkx/status_macros.h"

namespace zkx {

HloAliasAnalysis::HloAliasAnalysis(const HloModule* module) : module_(module) {}

const HloBuffer& HloAliasAnalysis::GetUniqueBufferAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  std::vector<const HloBuffer*> buffers = ComputeBuffersAt(instruction, index);
  CHECK_EQ(buffers.size(), 1);
  return *buffers[0];
}

HloBuffer& HloAliasAnalysis::GetUniqueBufferAt(
    const HloInstruction* instruction, const ShapeIndex& index) {
  return GetBuffer(const_cast<const HloAliasAnalysis*>(this)
                       ->GetUniqueBufferAt(instruction, index)
                       .id());
}

std::vector<const HloBuffer*> HloAliasAnalysis::ComputeBuffersAt(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  const HloValueSet& value_set =
      dataflow_analysis_->GetValueSet(instruction, index);
  std::vector<const HloBuffer*> buffers;
  buffers.reserve(value_set.values().size());
  for (const HloValue* value : value_set.values()) {
    buffers.push_back(&GetBufferContainingValue(*value));
  }

  // Sort and uniquify vector before returning.
  absl::c_sort(buffers, HloBuffer::IdLessThan);
  buffers.erase(std::unique(buffers.begin(), buffers.end()), buffers.end());

  return buffers;
}

absl::Status HloAliasAnalysis::Verify() const {
  // Verify consistency between the value_to_buffer_ map and
  // HloBuffer::values().
  for (const auto& pair : value_to_buffer_) {
    const HloValue* value = pair.first;
    const HloBuffer& buffer = *pair.second;
    TF_RET_CHECK(absl::c_linear_search(buffer.values(), value));
  }

  for (HloBuffer::Id id = 0; id < buffers_.size(); ++id) {
    const HloBuffer& buffer = buffers_[id];
    TF_RET_CHECK(buffer.id() == id);

    HloValue::Id last_value_id = -1;
    for (const HloValue* value : buffer.values()) {
      TF_RET_CHECK(GetBufferContainingValue(*value) == buffer);

      // Also verify the values in HloBuffer are unique and sorted by id.
      TF_RET_CHECK(value->id() > last_value_id);
      last_value_id = value->id();
    }
  }

  return absl::OkStatus();
}

std::string HloAliasAnalysis::ToString() const {
  std::string out =
      absl::StrCat("HloAliasAnalysis, module ", module_->name(), "\n");
  absl::StrAppend(&out, "  Buffers at each position:\n");
  for (const HloComputation* computation : module_->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      absl::StrAppend(&out, "    ", instruction->name(), ":\n");
      if (instruction->shape().IsTuple()) {
        ShapeUtil::ForEachSubshape(
            instruction->shape(),
            [&out, &instruction, this](const Shape&, const ShapeIndex& index) {
              absl::StrAppend(&out, "      tuple index ", index.ToString(),
                              ":\n");
              for (const HloBuffer* buffer :
                   ComputeBuffersAt(instruction, index)) {
                absl::StrAppend(&out, "        ", buffer->ToString(), "\n");
              }
            });
      } else {
        for (const HloBuffer* buffer :
             ComputeBuffersAt(instruction, /*index=*/{})) {
          absl::StrAppend(&out, "      ", buffer->ToString(), "\n");
        }
      }
    }
  }

  absl::StrAppend(&out, "  Buffers:\n");
  for (const HloBuffer& buffer : buffers()) {
    absl::StrAppend(&out, "    ", buffer.ToString(), "\n");
    absl::StrAppend(&out, "      positions:\n");
    for (const HloPosition& position : buffer.ComputePositions()) {
      absl::StrAppend(&out, "        ", position.ToString(), "\n");
    }
  }

  return out;
}

}  // namespace zkx
