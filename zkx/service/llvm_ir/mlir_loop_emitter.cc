#include "zkx/service/llvm_ir/mlir_loop_emitter.h"

#include "absl/log/check.h"
#include "mlir/IR/BuiltinTypes.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/shape_util.h"

namespace zkx::llvm_ir {

BodyEmitter MakeBodyEmitter(const ElementGenerator& target_element_generator,
                            absl::Span<const MlirArray> target_arrays,
                            EmitterLocOpBuilder& b, bool is_tuple) {
  // TODO(chokobole): Implement this.
  CHECK(!is_tuple) << "Not implemented";
  std::vector<MlirArray> target_arrays_vec(target_arrays.begin(),
                                           target_arrays.end());
  CHECK_EQ(target_arrays.size(), 1);
  return [=, &b](const MlirArray::Index array_index) -> absl::Status {
    // Convert `target_element_generator` to a BodyEmitter.
    TF_ASSIGN_OR_RETURN(mlir::Value target_element,
                        target_element_generator(array_index));
    target_arrays_vec[0].EmitWriteArrayElement(array_index, target_element, b);
    return absl::OkStatus();
  };
}

MlirLoopEmitter::MlirLoopEmitter(const BodyEmitter& body_emitter,
                                 const Shape& shape, EmitterLocOpBuilder& b)
    : body_emitter_(body_emitter), shape_(shape), b_(b) {}

MlirLoopEmitter::MlirLoopEmitter(const BodyEmitter& body_emitter,
                                 const Shape& shape,
                                 std::vector<mlir::Value> dynamic_dims,
                                 EmitterLocOpBuilder& b)
    : MlirLoopEmitter::MlirLoopEmitter(body_emitter, shape, b) {
  CHECK_EQ(dynamic_dims.size(), shape_.dimensions_size());
  dynamic_dims_ = std::move(dynamic_dims);
}

MlirLoopEmitter::MlirLoopEmitter(
    const ElementGenerator& target_element_generator,
    const MlirArray& target_array, EmitterLocOpBuilder& b)
    : body_emitter_(MakeBodyEmitter(target_element_generator, {target_array}, b,
                                    /*is_tuple=*/false)),
      shape_(target_array.GetShape()),
      b_(b) {}

MlirLoopEmitter::MlirLoopEmitter(
    const ElementGenerator& target_element_generator,
    absl::Span<const MlirArray> target_arrays, EmitterLocOpBuilder& b)
    : body_emitter_(MakeBodyEmitter(target_element_generator, target_arrays, b,
                                    /*is_tuple=*/true)),
      shape_(target_arrays[0].GetShape()),
      b_(b) {
  // Sanity check: In multi-output fusion, all shapes produced must have the
  // same dimensions.
  for (const MlirArray& array : target_arrays) {
    CHECK(ShapeUtil::SameDimensions(shape_, array.GetShape()))
        << ": '" << shape_.ShortDebugString() << "' does not match '"
        << array.GetShape().ShortDebugString() << "'";
  }
}

std::vector<MlirArray::Index> MlirLoopEmitter::EmitIndexAndSetExitBlock(
    std::string_view loop_name, mlir::Value base_index, mlir::Type index_type) {
  CHECK(index_type);
  CHECK(!base_index)
      << "ZKX CPU implementation of"
      << " MlirLoopEmitter::EmitIndexAndSetExitBlock doesn't support"
      << " base_index, but it was requested.";

  if (ShapeUtil::IsScalar(shape_)) {
    // No loop needed, so set `exit_block_` to empty.
    exit_block_ = nullptr;
    return {MlirArray::Index(index_type)};
  }

  return {MlirArray::Index(index_type)};
}

absl::Status MlirLoopEmitter::EmitLoop(std::string_view loop_name,
                                       mlir::Type index_type) {
  if (!index_type) {
    index_type = b_.getI64Type();
  }

  for (const MlirArray::Index& array_index :
       EmitIndexAndSetExitBlock(loop_name,
                                /*base_index=*/nullptr, index_type)) {
    TF_RETURN_IF_ERROR(body_emitter_(array_index));
  }
  return absl::OkStatus();
}

}  // namespace zkx::llvm_ir
