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

#include "zkx/service/gpu/ir_emitter.h"

#include "absl/log/log.h"

namespace zkx::gpu {

IrEmitter::IrEmitter(IrEmitterContext* ir_emitter_context, bool is_nested)
    : ir_emitter_context_(ir_emitter_context),
      module_(ir_emitter_context->llvm_module()),
      b_(module_->getContext())
// TODO(chokobole): Uncomment this. Dependency: HloToIrBindings.
// bindings_(&b_, module_, is_nested)
{}

absl::Status IrEmitter::DefaultAction(HloInstruction* hlo) {
  // ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  // for (const HloInstruction* operand : hlo->operands()) {
  //   operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index)
  //   {
  //     return GetIrArray(*operand, *hlo)
  //         .EmitReadArrayElement(index, &b_, operand->name());
  //   };
  // }
  // return EmitTargetElementLoop(
  //     *hlo, ElementalIrEmitter(module_, &b_)
  //               .MakeElementGenerator(hlo, operand_to_generator));
  return absl::UnimplementedError("DefaultAction is not implemented on GPU");
}

absl::Status IrEmitter::HandleConstant(HloInstruction* constant) {
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleAddDependency(HloInstruction* add_dependency) {
  VLOG(2) << "HandleAddDependency: " << add_dependency->ToString();
  // TODO(chokobole): Uncomment this. Dependency: HloToIrBindings.
  // const HloInstruction* operand = add_dependency->operand(0);
  // // Add_Dependency is a no-op, but we still want to bind it to an
  // llvm::Value
  // // sometimes, e.g., when it's operand is a constant or a bitcast of a
  // // constant.
  // if (bindings_.BoundToIrValue(*operand)) {
  //   bindings_.BindHloToIrValue(*add_dependency, GetBasePointer(*operand));
  // }
  // return absl::OkStatus();
  return absl::UnimplementedError("AddDependency is not implemented on GPU");
}

absl::Status IrEmitter::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  // TODO(chokobole): Uncomment this. Dependency: HloToIrBindings.
  // auto operand = get_tuple_element->operand(0);
  // CHECK(bindings_.BoundToIrValue(*operand));
  // bindings_.BindHloToIrValue(
  //     *get_tuple_element,
  //     llvm_ir::EmitGetTupleElement(
  //         get_tuple_element->shape(), get_tuple_element->tuple_index(),
  //         // TODO(b/26344050): tighten the alignment here
  //         // based on the real element type.
  //         /*alignment=*/1, GetBasePointer(*operand),
  //         llvm_ir::ShapeToIrType(operand->shape(), module_->getContext()),
  //         &b_));
  // return absl::OkStatus();
  return absl::UnimplementedError("GetTupleElement is not implemented on GPU");
}

absl::Status IrEmitter::HandleSend(HloInstruction*) {
  return absl::UnimplementedError("Send is not implemented on GPU");
}

absl::Status IrEmitter::HandleSendDone(HloInstruction*) {
  return absl::UnimplementedError("Send-Done is not implemented on GPU");
}

absl::Status IrEmitter::HandleRecv(HloInstruction*) {
  return absl::UnimplementedError("Recv is not implemented on GPU");
}

absl::Status IrEmitter::HandleRecvDone(HloInstruction*) {
  return absl::UnimplementedError("Recv-done is not implemented on GPU");
}

absl::Status IrEmitter::HandleScatter(HloInstruction*) {
  return absl::UnimplementedError("Scatter is not implemented on GPUs.");
}

absl::Status IrEmitter::HandleTuple(HloInstruction* tuple) {
  // TODO(chokobole): Uncomment this. Dependency: EmitTuple.
  // std::vector<llvm::Value*> base_ptrs;
  // for (const HloInstruction* operand : tuple->operands()) {
  //   base_ptrs.push_back(GetBasePointer(*operand));
  // }
  // llvm_ir::EmitTuple(GetIrArray(*tuple, *tuple), base_ptrs, &b_);
  // return absl::OkStatus();
  return absl::UnimplementedError("Tuple is not implemented on GPUs.");
}

absl::Status IrEmitter::HandleAllReduce(HloInstruction* crs) {
  return absl::UnimplementedError(
      "AllReduce cannot be nested inside of fusion, map, etc.");
}

absl::Status IrEmitter::HandleParameter(HloInstruction* parameter) {
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleCall(HloInstruction* call) {
  // TODO(chokobole): Uncomment this. Dependency: CallNestedComputation.
  // std::vector<llvm::Value*> operand_addresses;
  // for (HloInstruction* operand : call->operands()) {
  //   operand_addresses.push_back(GetBasePointer(*operand));
  // }
  // return CallNestedComputation(&b_, *ir_emitter_context_, *call->to_apply(),
  //                              operand_addresses, GetBasePointer(*call));
  return absl::UnimplementedError("Call is not implemented on GPU");
}

absl::Status IrEmitter::HandleCustomCall(HloInstruction*) {
  return absl::UnimplementedError("custom-call");
}

absl::Status IrEmitter::HandleInfeed(HloInstruction*) {
  // TODO(b/30467474): Implement infeed on GPU.
  return absl::UnimplementedError("Infeed is not supported on GPU.");
}

absl::Status IrEmitter::HandleOutfeed(HloInstruction*) {
  // TODO(b/34359662): Implement outfeed on GPU.
  return absl::UnimplementedError("Outfeed is not supported on GPU.");
}

// TODO(chokobole): Uncomment this. Dependency: HloToIrBindings.
// std::vector<llvm_ir::IrArray> IrEmitter::ConstructIrArrayForOutputs(
//     const HloInstruction& hlo) {
//   std::vector<llvm_ir::IrArray> output_arrays;
//   if (hlo.shape().IsTuple()) {
//     int64_t num_outputs = ShapeUtil::TupleElementCount(hlo.shape());
//     output_arrays.reserve(num_outputs);
//     for (int64_t i = 0; i < num_outputs; ++i) {
//       output_arrays.push_back(GetIrArray(hlo, hlo, {i}));
//     }
//   } else {
//     output_arrays.push_back(GetIrArray(hlo, hlo));
//   }
//   return output_arrays;
// }

// TODO(chokobole): Uncomment this. Dependency: FusedIrEmitter.
// void IrEmitter::BindFusionArguments(const HloInstruction* fusion,
//                                     FusedIrEmitter* fused_emitter) {
//   for (int i = 0; i < fusion->operand_count(); i++) {
//     const HloInstruction* operand = fusion->operand(i);
//     fused_emitter->BindGenerator(
//         *fusion->fused_parameter(i),
//         [this, operand, fusion](llvm_ir::IrArray::Index index) {
//           return GetIrArray(*operand, *fusion)
//               .EmitReadArrayElement(index, &b_, operand->name());
//         });
//   }
// }

}  // namespace zkx::gpu
