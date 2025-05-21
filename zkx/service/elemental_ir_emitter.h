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

#ifndef ZKX_SERVICE_ELEMENTAL_IR_EMITTER_H_
#define ZKX_SERVICE_ELEMENTAL_IR_EMITTER_H_

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include "zkx/codegen/emitter_loc_op_builder.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/service/llvm_ir/mlir_loop_emitter.h"

namespace zkx {

class ElementalIrEmitter {
 public:
  using HloToElementGeneratorMap =
      absl::flat_hash_map<const HloInstruction*, llvm_ir::ElementGenerator>;

  explicit ElementalIrEmitter(EmitterLocOpBuilder& b) : b_(b) {}

  virtual ~ElementalIrEmitter() = default;

  // Returns a function to generate an element of the output of `hlo`, given a
  // map of functions to generate elements of its operands.
  llvm_ir::ElementGenerator MakeElementGenerator(
      const HloInstruction* hlo,
      const HloToElementGeneratorMap& operand_to_generator);

  EmitterLocOpBuilder& b() { return b_; }

 protected:
  virtual absl::StatusOr<mlir::Value> EmitBinaryOp(const HloInstruction* op,
                                                   mlir::Value lhs_value,
                                                   mlir::Value rhs_value);

  virtual absl::StatusOr<mlir::Value> EmitIntegerBinaryOp(
      const HloInstruction* op, mlir::Value lhs_value, mlir::Value rhs_value,
      bool is_signed);
  virtual absl::StatusOr<mlir::Value> EmitPrimeFieldBinaryOp(
      const HloInstruction* op, mlir::Value lhs_value, mlir::Value rhs_value);

  EmitterLocOpBuilder& b_;
};

}  // namespace zkx

#endif  // ZKX_SERVICE_ELEMENTAL_IR_EMITTER_H_
