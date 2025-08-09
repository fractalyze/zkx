#include "zkx/service/gpu/ir_emission_utils.h"

namespace zkx::gpu {

absl::StatusOr<BufferAllocation::Slice> GetAllocationSlice(
    const BufferAssignment& buffer_assignment, const HloInstruction* instr,
    const ShapeIndex& index) {
  return buffer_assignment.GetUniqueSlice(instr, index);
}

}  // namespace zkx::gpu
