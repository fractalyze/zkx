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

#include "zkx/service/buffer_assignment.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#include "zkx/base/logging.h"
#include "zkx/map_util.h"

namespace zkx {

std::string BufferAllocation::Slice::ToString() const {
  return absl::StrCat("{index:", allocation_ == nullptr ? -1 : index(),
                      ", offset:", offset_, ", size:", size_, "}");
}

BufferAllocation::Slice BufferAllocation::GetSlice(
    const HloValue& buffer) const {
  const OffsetSize os = FindOrDie(assigned_buffers_, &buffer);
  return Slice(this, os.offset, os.size);
}

std::ostream& operator<<(std::ostream& out, const BufferAllocation::Slice& s) {
  out << s.ToString();
  return out;
}

bool BufferAssignment::HasAllocation(const HloValue& value) const {
  return allocation_index_for_value_.contains(&value);
}

bool BufferAssignment::HasAllocation(HloValue::Id value_id) const {
  return HasAllocation(dataflow_analysis().GetValue(value_id));
}

bool BufferAssignment::HasAllocation(const HloBuffer& buffer) const {
  return allocation_index_for_value_.contains(buffer.values()[0]);
}

const BufferAllocation& BufferAssignment::GetAssignedAllocation(
    const HloValue& value) const {
  CHECK(HasAllocation(value));
  return GetAllocation(allocation_index_for_value_.at(&value));
}

const BufferAllocation& BufferAssignment::GetAssignedAllocation(
    const HloBuffer& hlo_buffer) const {
  return GetAssignedAllocation(*hlo_buffer.values()[0]);
}

const BufferAllocation& BufferAssignment::GetAllocation(
    BufferAllocation::Index index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, allocations_.size());
  return allocations_[index];
}

absl::StatusOr<BufferAllocation::Slice> BufferAssignment::GetUniqueSlice(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  VLOG(3) << "Trying to find unique slice for " << instruction->name() << " ["
          << index << "]";
  BufferAllocation::Slice result;
  for (const HloValue* value :
       dataflow_analysis().GetValueSet(instruction, index).values()) {
    VLOG(3) << "Examining value " << *value;
    if (HasAllocation(*value)) {
      VLOG(3) << "Has allocation";
      const BufferAllocation::Slice slice =
          GetAssignedAllocation(*value).GetSlice(*value);
      if (result.allocation() == nullptr) {
        result = slice;
      } else if (result != slice) {
        return absl::FailedPreconditionError(absl::StrFormat(
            "BufferAllocation::Slice for instruction %s at index %s cannot "
            "be determined at compile-time.",
            instruction->name(), index.ToString()));
      }
    } else {
      VLOG(3) << "No allocation";
    }
  }
  if (result.allocation() == nullptr) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "BufferAllocation::Slice not assigned for instruction %s at index %s",
        instruction->name(), index.ToString()));
  }
  return result;
}

absl::StatusOr<BufferAllocation::Slice>
BufferAssignment::GetUniqueTopLevelSlice(
    const HloInstruction* instruction) const {
  return GetUniqueSlice(instruction, /*index=*/{});
}

}  // namespace zkx
