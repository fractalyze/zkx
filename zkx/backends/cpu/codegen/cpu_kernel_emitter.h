/* Copyright 2024 The OpenXLA Authors.

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

#ifndef ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_H_
#define ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"

#include "zkx/codegen/emitter_loc_op_builder.h"
#include "zkx/codegen/kernel_emitter.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/service/buffer_assignment.h"

namespace zkx::cpu {

class CpuKernelEmitter final : public KernelEmitter {
 public:
  CpuKernelEmitter(mlir::MLIRContext* context, const HloInstruction* instr,
                   const BufferAssignment* buffer_assignment);

  absl::StatusOr<KernelDefinition> EmitKernelDefinition() override;

 private:
  absl::StatusOr<llvm::SmallVector<mlir::Type>> MakeFuncArguments() const;

  absl::StatusOr<absl::flat_hash_map<const HloInstruction*, mlir::Value>>
  EmitOperands(EmitterLocOpBuilder& b, mlir::Block* entry_block) const;

  mlir::Value EmitCSROperand(EmitterLocOpBuilder& b, mlir::Block* entry_block,
                             int64_t i, const Shape& shape) const;

  mlir::Value EmitCSCOperand(EmitterLocOpBuilder& b, mlir::Block* entry_block,
                             int64_t i, const Shape& shape) const;

  mlir::Value EmitCOOOperand(EmitterLocOpBuilder& b, mlir::Block* entry_block,
                             int64_t i, const Shape& shape) const;

  absl::Status EmitEpilog(EmitterLocOpBuilder& b, mlir::Block* entry_block,
                          mlir::Value res) const;

  absl::StatusOr<mlir::Value> EmitOp(
      const HloInstruction* instr, EmitterLocOpBuilder& b,
      absl::flat_hash_map<const HloInstruction*, mlir::Value>& values);

  absl::StatusOr<mlir::Value> EmitUnaryOp(const HloInstruction* instr,
                                          EmitterLocOpBuilder& b,
                                          mlir::Value value);

  absl::StatusOr<mlir::Value> EmitBinaryOp(const HloInstruction* instr,
                                           EmitterLocOpBuilder& b,
                                           mlir::Value lhs_value,
                                           mlir::Value rhs_value);

  absl::StatusOr<mlir::Value> EmitFftOp(const HloInstruction* instr,
                                        EmitterLocOpBuilder& b,
                                        mlir::Value value);

  absl::StatusOr<mlir::Value> EmitMsmOp(const HloInstruction* instr,
                                        EmitterLocOpBuilder& b,
                                        mlir::Value scalars, mlir::Value bases);

  absl::StatusOr<mlir::Value> EmitDimensionsOp(
      const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value input,
      absl::Span<const int64_t> source_dimensions);

  absl::StatusOr<mlir::Value> EmitDotOp(const HloInstruction* instr,
                                        EmitterLocOpBuilder& b, mlir::Value lhs,
                                        mlir::Value rhs);

  absl::StatusOr<mlir::Value> EmitSliceOp(
      const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value,
      absl::Span<const int64_t> start_indices,
      absl::Span<const int64_t> limit_indices,
      absl::Span<const int64_t> strides);

  absl::StatusOr<mlir::Value> EmitIntegerUnaryOp(const HloInstruction* instr,
                                                 EmitterLocOpBuilder& b,
                                                 mlir::Value value);

  absl::StatusOr<mlir::Value> EmitFieldUnaryOp(const HloInstruction* instr,
                                               EmitterLocOpBuilder& b,
                                               mlir::Value value);

  absl::StatusOr<mlir::Value> EmitEcPointUnaryOp(const HloInstruction* instr,
                                                 EmitterLocOpBuilder& b,
                                                 mlir::Value value);

  absl::StatusOr<mlir::Value> EmitIntegerBinaryOp(const HloInstruction* instr,
                                                  EmitterLocOpBuilder& b,
                                                  mlir::Value lhs_value,
                                                  mlir::Value rhs_value,
                                                  bool is_signed);

  absl::StatusOr<mlir::Value> EmitFieldBinaryOp(const HloInstruction* instr,
                                                EmitterLocOpBuilder& b,
                                                mlir::Value lhs_value,
                                                mlir::Value rhs_value);

  absl::StatusOr<mlir::Value> EmitEcPointBinaryOp(const HloInstruction* instr,
                                                  EmitterLocOpBuilder& b,
                                                  mlir::Value lhs_value,
                                                  mlir::Value rhs_value);

  absl::StatusOr<mlir::Value> EmitMatrixVectorMultiplicationOp(
      const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs,
      mlir::Value rhs);

  mlir::MLIRContext* const mlir_context_;
  const HloInstruction* const instr_;

  const BufferAssignment* const buffer_assignment_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_H_
