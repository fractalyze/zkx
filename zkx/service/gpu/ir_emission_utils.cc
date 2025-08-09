#include "zkx/service/gpu/ir_emission_utils.h"

#include "zkx/primitive_util.h"
#include "zkx/util.h"

namespace zkx::gpu {

absl::StatusOr<BufferAllocation::Slice> GetAllocationSlice(
    const BufferAssignment& buffer_assignment, const HloInstruction* instr,
    const ShapeIndex& index) {
  return buffer_assignment.GetUniqueSlice(instr, index);
}

absl::StatusOr<DenseDataIntermediate> LiteralToZkxFormat(
    const Literal& literal) {
  PrimitiveType element_type = literal.shape().element_type();
  if (!primitive_util::IsArrayType(element_type)) {
    return absl::InternalError("Unsupported type in LiteralToXlaFormat");
  }

  int64_t byte_size = literal.size_bytes();
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    auto bit_width = primitive_util::BitWidth(element_type);
    std::vector<uint8_t> output(CeilOfRatio<int64_t>(byte_size, 8 / bit_width));
    absl::Span<char> output_span =
        absl::MakeSpan(reinterpret_cast<char*>(output.data()), output.size());
    PackIntN(
        bit_width,
        absl::MakeSpan(reinterpret_cast<const char*>(literal.untyped_data()),
                       byte_size),
        output_span);
    return DenseDataIntermediate::Own(std::move(output));
  }

  return DenseDataIntermediate::Alias(absl::MakeSpan(
      reinterpret_cast<const uint8_t*>(literal.untyped_data()), byte_size));
}

}  // namespace zkx::gpu
