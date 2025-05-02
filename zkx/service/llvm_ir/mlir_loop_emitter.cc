#include "zkx/service/llvm_ir/mlir_loop_emitter.h"

#include <memory>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/base/logging.h"
#include "zkx/layout_util.h"
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

MlirArray::Index MlirLoopEmitter::EmitStaticIndex(MlirForLoopNest* loop_nest,
                                                  mlir::Type index_type) {
  // Create loop nest with one for-loop for each dimension of the target shape.
  // Loops are added from outermost to innermost order with the MlirForLoopNest
  // class so emit loops in order from most-major dimension down to most-minor
  // dimension (of the target shape).
  std::vector<mlir::Value> array_multi_index(shape_.dimensions_size());
  for (int i = 0; i < LayoutUtil::MinorToMajor(shape_).size(); ++i) {
    int64_t dimension = LayoutUtil::Major(shape_.layout(), i);
    // Only unroll the most minor dimension, this seems to give us good runtime
    // performance with a large improvement in compile time.
    auto unroll_mode = (i == shape_.rank() - 1) ? UnrollMode::kDefaultUnroll
                                                : UnrollMode::kNoUnroll;
    if (i > 0) {
      // Create this loop inside the previous one.
      b_.setInsertionPointToStart(loop_nest->GetInnerLoopBodyBlock());
    }
    std::unique_ptr<MlirForLoop> loop = loop_nest->AddLoop(
        /*start_index=*/0,
        /*end_index=*/shape_.dimensions(dimension),
        /*suffix=*/absl::StrFormat("dim.%d", dimension), unroll_mode);
    array_multi_index[dimension] = loop->GetIndVarValue();
  }
  return MlirArray::Index(array_multi_index, shape_, index_type);
}

MlirArray::Index MlirLoopEmitter::EmitDynamicIndex(MlirForLoopNest* loop_nest,
                                                   mlir::Type index_type) {
  CHECK_EQ(shape_.is_dynamic(), true);
  // Create loop nest with one for-loop for each dynamic dimensions.
  // Loops are added from outermost to innermost order with the MlirForLoopNest
  // class so emit loops in order from most-major dimension down to most-minor
  // dimension (of the target shape).
  std::vector<mlir::Value> array_multi_index(shape_.dimensions_size());
  for (int i = 0; i < LayoutUtil::MinorToMajor(shape_).size(); ++i) {
    int64_t dimension = LayoutUtil::Major(shape_.layout(), i);
    // Only unroll the most minor dimension, this seems to give us good runtime
    // performance with a large improvement in compile time.
    auto unroll_mode = (i == shape_.rank() - 1) ? UnrollMode::kDefaultUnroll
                                                : UnrollMode::kNoUnroll;
    std::unique_ptr<MlirForLoop> loop = loop_nest->AddLoop(
        /*suffix=*/absl::StrFormat("dim.%d", dimension),
        /*start_index=*/
        b_.create<mlir::LLVM::ConstantOp>(index_type,
                                          b_.getIntegerAttr(index_type, 0)),
        /*end_index=*/dynamic_dims_[dimension], unroll_mode);
    array_multi_index[dimension] = loop->GetIndVarValue();
  }
  return MlirArray::Index(array_multi_index, shape_, index_type);
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

  MlirForLoopNest loop_nest(loop_name, b_);

  MlirArray::Index array_index = dynamic_dims_.empty()
                                     ? EmitStaticIndex(&loop_nest, index_type)
                                     : EmitDynamicIndex(&loop_nest, index_type);

  // Set IR builder insertion point to the loop body block of the
  // innermost loop.
  mlir::Block* innermost_body_bb = loop_nest.GetInnerLoopBodyBlock();
  b_.setInsertionPointToStart(innermost_body_bb);

  // Set `exit_block_` to the exit block of the loop nest.
  exit_block_ = loop_nest.GetOuterLoopExitBlock();
  CHECK_NOTNULL(exit_block_);

  return {array_index};
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

  // Set the insertion point of `b_` to the loop exit, so that
  // code emitted for later instructions will be correctly placed.
  if (exit_block_ != nullptr) {
    b_.setInsertionPointToEnd(exit_block_);
  }
  return absl::OkStatus();
}

}  // namespace zkx::llvm_ir
