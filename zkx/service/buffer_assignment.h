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

#ifndef ZKX_SERVICE_BUFFER_ASSIGNMENT_H_
#define ZKX_SERVICE_BUFFER_ASSIGNMENT_H_

#include <stdint.h>

#include <utility>

#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"

#include "zkx/hlo/analysis/hlo_alias_analysis.h"
#include "zkx/service/hlo_buffer.h"
#include "zkx/service/hlo_value.h"
#include "zkx/service/logical_buffer.h"

namespace zkx {

// This class abstracts an allocation of contiguous memory which can hold the
// values described by LogicalBuffers. Each LogicalBuffer occupies a sub-range
// of the allocation, represented by a Slice. A single BufferAllocation may hold
// LogicalBuffers with disjoint liveness, which may have overlapping Slices. A
// single BufferAllocation may also hold LogicalBuffers with overlapping
// liveness, which must have disjoint Slices.
//
// The abstraction includes information required by the backends for allocation,
// use, and deallocation of the buffer. This includes the LogicalBuffers which
// are held in this allocation through the execution of the computation.
class BufferAllocation {
 public:
  // Holds a unique identifier for each allocation. Values are assigned
  // contiguously and can be used as array indexes.
  using Index = int64_t;

  BufferAllocation(Index index, int64_t size, LogicalBuffer::Color color)
      : index_(index), size_(size), color_(color) {}

  // Returns the index of this allocation.
  Index index() const { return index_; }

  // Returns the size of the allocation. Necessarily this must be at least as
  // large as any LogicalBuffer assigned to this allocation.
  int64_t size() const { return size_; }

  // Returns the color of the allocation. Only logical buffers with a matching
  // color can reside in this allocation.
  LogicalBuffer::Color color() const { return color_; }

  void set_color(LogicalBuffer::Color color) { color_ = color; }

  struct OffsetSize {
    int64_t offset = 0;
    int64_t size = 0;
  };

  // Access to the logical buffers assigned to this allocation, and their
  // associated logical offsets and sizes.
  const absl::flat_hash_map<const HloValue*, OffsetSize>& assigned_buffers()
      const {
    return assigned_buffers_;
  }

  // A Slice represents a contiguous portion of a memory allocation. It is used
  // to identify the memory range that a LogicalBuffer corresponds to.
  class Slice {
   public:
    Slice() = default;
    Slice(const BufferAllocation* allocation, int64_t offset, int64_t size)
        : allocation_(allocation), offset_(offset), size_(size) {}

    const BufferAllocation* allocation() const { return allocation_; }
    Index index() const { return allocation_->index(); }
    int64_t offset() const { return offset_; }
    int64_t size() const { return size_; }

    bool operator==(const Slice& other) const {
      return index() == other.index() && offset_ == other.offset_ &&
             size_ == other.size_;
    }
    bool operator!=(const Slice& other) const { return !(*this == other); }
    bool operator<(const Slice& other) const {
      if (index() != other.index()) return index() < other.index();
      if (offset_ != other.offset_) return offset_ < other.offset_;
      return size_ < other.size_;
    }

    // Returns true iff this slice's memory range has a non-empty intersection
    // with the other slice's memory range.
    bool OverlapsWith(const Slice& other) const {
      const int64_t end = offset_ + size_;
      const int64_t other_end = other.offset_ + other.size_;
      return index() == other.index() && offset_ < other_end &&
             end > other.offset_;
    }

    template <typename H>
    friend H AbslHashValue(H h, const Slice& s) {
      return H::combine(std::move(h), s.index(), s.offset(), s.size());
    }

    std::string ToString() const;

   private:
    const BufferAllocation* allocation_ = nullptr;
    int64_t offset_ = 0;
    int64_t size_ = 0;
  };

  // GetSlice returns the Slice of contiguous memory that holds the value
  // described by the given 'buffer'.
  // REQUIRES: 'buffer' must be assigned to this allocation.
  Slice GetSlice(const HloValue& buffer) const;

 private:
  // The index of the allocation in the BufferAssignment.
  Index index_;

  // Size of the allocation in bytes.
  int64_t size_;

  // Color of the allocation.
  LogicalBuffer::Color color_;

  // Mapping from the set of buffers assigned to this allocation to their
  // logical offsets and sizes.
  absl::flat_hash_map<const HloValue*, OffsetSize> assigned_buffers_;
};

// Add stream operators for nicer output of CHECK/RET_CHECK failures.
std::ostream& operator<<(std::ostream& out, const BufferAllocation::Slice& s);

// This class encapsulates an assignment of the LogicalBuffers in a ZKX
// module to a set of BufferAllocations.
class BufferAssignment {
 public:
  // Returns the vector containing all buffer allocations in this assignment.
  const std::vector<BufferAllocation>& Allocations() const {
    return allocations_;
  }

  // Returns whether the given buffer has been assigned an allocation.
  bool HasAllocation(const HloValue& value) const;

  // Returns whether the given (logical) buffer with the id has been assigned an
  // allocation.
  bool HasAllocation(HloValue::Id value_id) const;

  bool HasAllocation(const HloBuffer& buffer) const;

  // Returns the allocation that a particular LogicalBuffer has been assigned
  // to. CHECKs if buffer has not been assigned an allocation.
  const BufferAllocation& GetAssignedAllocation(const HloValue& value) const;

  const BufferAllocation& GetAssignedAllocation(
      const HloBuffer& hlo_buffer) const;

  // Returns the allocation with the given index. CHECKs if no allocation exists
  // with the given index.
  const BufferAllocation& GetAllocation(BufferAllocation::Index index) const;

  // Convenience function which returns the unique slice containing the buffer
  // at the given index of the given instruction. If a slice is not assigned or
  // the slice cannot be determined at compile time then an error is returned.
  absl::StatusOr<BufferAllocation::Slice> GetUniqueSlice(
      const HloInstruction* instruction, const ShapeIndex& index) const;
  // Like GetUniqueSlice but fixes the index to the top-level of the shape
  // (index = {}).
  absl::StatusOr<BufferAllocation::Slice> GetUniqueTopLevelSlice(
      const HloInstruction* instruction) const;

  // Returns the set BufferValues which may be the source of the value at the
  // given index and instruction.
  const std::vector<const HloValue*>& GetSourceBuffers(
      const HloInstruction* instruction, const ShapeIndex& index) const {
    return dataflow_analysis().GetValueSet(instruction, index).values();
  }

  const HloDataflowAnalysis& dataflow_analysis() const {
    return alias_analysis_->dataflow_analysis();
  }

  HloAliasAnalysis& alias_analysis() const { return *alias_analysis_; }

 private:
  // The vector of buffer allocations. Indexed by BufferAllocation::Index.
  std::vector<BufferAllocation> allocations_;

  // Maps Buffers to the index of the BufferAllocation which holds the buffer.
  absl::flat_hash_map<const HloValue*, BufferAllocation::Index>
      allocation_index_for_value_;

  std::unique_ptr<HloAliasAnalysis> alias_analysis_;

  BufferAssignment(const BufferAssignment&) = delete;
  BufferAssignment& operator=(const BufferAssignment&) = delete;
};

}  // namespace zkx

#endif  // ZKX_SERVICE_BUFFER_ASSIGNMENT_H_
