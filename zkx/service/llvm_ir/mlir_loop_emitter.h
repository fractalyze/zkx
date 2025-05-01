#ifndef ZKX_SERVICE_LLVM_IR_LOOP_MLIR_EMITTER_H_
#define ZKX_SERVICE_LLVM_IR_LOOP_MLIR_EMITTER_H_

#include <functional>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

#include "zkx/codegen/emitter_loc_op_builder.h"
#include "zkx/service/llvm_ir/mlir_array.h"
#include "zkx/shape.h"

namespace zkx::llvm_ir {

using ElementGenerator =
    std::function<absl::StatusOr<mlir::Value>(const MlirArray::Index& index)>;
using BodyEmitter = std::function<absl::Status(const MlirArray::Index& index)>;

// Creates the body emitter from target arrays.
BodyEmitter MakeBodyEmitter(const ElementGenerator& target_element_generator,
                            absl::Span<const MlirArray> target_arrays,
                            EmitterLocOpBuilder& b, bool is_tuple);

// Emits a loop for every element in the given shape.
class MlirLoopEmitter {
 public:
  MlirLoopEmitter(const BodyEmitter& body_emitter, const Shape& shape,
                  EmitterLocOpBuilder& b);

  // Constructs a MlirLoopEmitter from an body_emitter that generates
  // element of the given target array in the dynamic dimension.
  MlirLoopEmitter(const BodyEmitter& body_emitter, const Shape& shape,
                  std::vector<mlir::Value> dynamic_dims,
                  EmitterLocOpBuilder& b);

  // Constructs a MlirLoopEmitter from an element generator that generates each
  // element of the given target array.
  MlirLoopEmitter(const ElementGenerator& target_element_generator,
                  const MlirArray& target_array, EmitterLocOpBuilder& b);

  // Constructs a MlirLoopEmitter that emits one element into each of N separate
  // arrays on each iteration of the loop.
  //
  // This is used for multi-output fusion.  target_element_generator must
  // produce an LLVM struct with N elements.
  MlirLoopEmitter(const ElementGenerator& target_element_generator,
                  absl::Span<const MlirArray> target_arrays,
                  EmitterLocOpBuilder& b);

  MlirLoopEmitter(const MlirLoopEmitter&) = delete;
  MlirLoopEmitter& operator=(const MlirLoopEmitter&) = delete;
  virtual ~MlirLoopEmitter() = default;

  // Emits a loop nest (with a yet-to-be-filled loop body) that iterates through
  // every element in the given shape. Returns the multi-dimensional index that
  // specifies the element, will return multiple indices if the loop is
  // unrolled.
  virtual std::vector<MlirArray::Index> EmitIndexAndSetExitBlock(
      std::string_view loop_name, mlir::Value base_index,
      mlir::Type index_type);

  // Emits a complete loop nest for every element in the given shape.
  absl::Status EmitLoop(std::string_view loop_name = "",
                        mlir::Type index_type = nullptr);

 protected:
  // An IR emitter that generates the loop body.
  BodyEmitter body_emitter_;

  // The shape that the emitted loop iterates through.
  Shape shape_;

  // Dynamic dimensions that emitted loop iterates through. Generate the
  // loop based on the dynamic dimensions if this vector is not empty.
  std::vector<mlir::Value> dynamic_dims_;

  // Points to the exit block of the emitted loop. If the given shape is
  // scalar, no loops are emitted and `exit_block_` is nullptr in that case.
  mlir::Block* exit_block_;

  EmitterLocOpBuilder& b_;
};

}  // namespace zkx::llvm_ir

#endif  // ZKX_SERVICE_LLVM_IR_LOOP_MLIR_EMITTER_H_
