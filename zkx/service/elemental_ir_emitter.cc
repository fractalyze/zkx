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

#include "zkx/service/elemental_ir_emitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "xla/tsl/platform/statusor.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"

namespace zkx {

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitIntegerBinaryOp(
    const HloInstruction* op, mlir::Value lhs_value, mlir::Value rhs_value,
    bool is_signed) {
  switch (op->opcode()) {
    // TODO(jingyue): add the "nsw" attribute for signed types.
    case HloOpcode::kAdd:
      return b_.create<mlir::arith::AddIOp>(lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return b_.create<mlir::arith::SubIOp>(lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return b_.create<mlir::arith::MulIOp>(lhs_value, rhs_value);

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "binary integer op '%s'", HloOpcodeString(op->opcode())));
  }
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitFieldBinaryOp(
    const HloInstruction* op, mlir::Value lhs_value, mlir::Value rhs_value) {
  switch (op->opcode()) {
    case HloOpcode::kAdd:
      return b_.create<mlir::zkir::field::AddOp>(lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return b_.create<mlir::zkir::field::SubOp>(lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return b_.create<mlir::zkir::field::MulOp>(lhs_value, rhs_value);

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "binary field op '%s'", HloOpcodeString(op->opcode())));
  }
}

absl::StatusOr<mlir::Value> ElementalIrEmitter::EmitBinaryOp(
    const HloInstruction* op, mlir::Value lhs_value, mlir::Value rhs_value) {
  PrimitiveType operand_type = op->operand(0)->shape().element_type();
  if (ShapeUtil::ElementIsIntegral(op->operand(0)->shape())) {
    return EmitIntegerBinaryOp(
        op, lhs_value, rhs_value,
        primitive_util::IsSignedIntegralType(operand_type));
  } else if (ShapeUtil::ElementIsField(op->operand(0)->shape())) {
    return EmitFieldBinaryOp(op, lhs_value, rhs_value);
  }
  return absl::UnimplementedError("...");
}

llvm_ir::ElementGenerator ElementalIrEmitter::MakeElementGenerator(
    const HloInstruction* hlo,
    const ElementalIrEmitter::HloToElementGeneratorMap& operand_to_generator) {
  switch (hlo->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
      return [this, hlo,
              &operand_to_generator](const llvm_ir::MlirArray::Index& index)
                 -> absl::StatusOr<mlir::Value> {
        const HloInstruction* lhs = hlo->operand(0);
        const HloInstruction* rhs = hlo->operand(1);
        TF_ASSIGN_OR_RETURN(mlir::Value lhs_value,
                            operand_to_generator.at(lhs)(index));
        TF_ASSIGN_OR_RETURN(mlir::Value rhs_value,
                            operand_to_generator.at(rhs)(index));
        return EmitBinaryOp(hlo, lhs_value, rhs_value);
      };
    default:
      return [hlo](const llvm_ir::MlirArray::Index& index) {
        return absl::UnimplementedError(
            absl::StrFormat("Unhandled opcode for elemental IR emission: %s",
                            HloOpcodeString(hlo->opcode())));
      };
  }
}

}  // namespace zkx
