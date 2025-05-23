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

#include "zkx/backends/cpu/codegen/elemental/elemental_kernel_emitter.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Linker/Linker.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "xla/tsl/platform/statusor.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkx/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "zkx/base/logging.h"
#include "zkx/codegen/emitter_loc_op_builder.h"
#include "zkx/codegen/llvm_ir_kernel_source.h"
#include "zkx/primitive_util.h"
#include "zkx/service/llvm_ir/llvm_util.h"
#include "zkx/shape_util.h"

namespace zkx::cpu {

namespace {

void LoadMlirDialects(mlir::MLIRContext* mlir_context) {
  mlir_context->loadDialect<
      // clang-format off
      mlir::arith::ArithDialect,
      mlir::bufferization::BufferizationDialect,
      mlir::cf::ControlFlowDialect,
      mlir::func::FuncDialect,
      mlir::LLVM::LLVMDialect,
      mlir::memref::MemRefDialect,
      mlir::zkir::elliptic_curve::EllipticCurveDialect,
      mlir::zkir::field::FieldDialect,
      mlir::zkir::mod_arith::ModArithDialect
      // clang-format on
      >();
}

void OneShotBufferize(mlir::OpPassManager& pm) {
  // NOTE: One-shot bufferize does not deallocate buffers. This is done by the
  // ownership-based buffer deallocation pass.
  // https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/
  mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  pm.addPass(mlir::memref::createExpandReallocPass());
  pm.addPass(mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(mlir::bufferization::createLowerDeallocationsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createBufferizationToMemRefPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void AddPasses(mlir::OpPassManager& pm) {
  pm.addPass(mlir::zkir::elliptic_curve::createEllipticCurveToField());
  pm.addPass(mlir::zkir::field::createFieldToModArith());
  pm.addPass(mlir::zkir::mod_arith::createModArithToArith());

  pm.addPass(mlir::createLowerAffinePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertElementwiseToLinalgPass());

  OneShotBufferize(pm);

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertLinalgToParallelLoopsPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertToLLVMPass());
}

std::unique_ptr<llvm::Module> TranslateMLIRToLLVM(
    mlir::ModuleOp module, llvm::LLVMContext* llvm_context) {
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(module, *llvm_context);
  CHECK(llvm_module) << "Failed to translate module to LLVM IR.";
  VLOG(2) << "LLVM module after translation";
  ZKX_VLOG_LINES(2, llvm_ir::DumpToString(llvm_module.get()));
  return llvm_module;
}

std::unique_ptr<llvm::Module> CreateLLVMModule(
    mlir::MLIRContext* mlir_context, mlir::ModuleOp module,
    llvm::LLVMContext* llvm_context) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerAllExtensions(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  module->getContext()->appendDialectRegistry(registry);

  VLOG(2) << "MLIR before optimizations";
  ZKX_VLOG_LINES(2, llvm_ir::DumpToString(module));
  mlir::PassManager pm(mlir_context);
  AddPasses(pm);

  CHECK(mlir::succeeded(pm.run(module)));

  VLOG(2) << "MLIR after optimizations";
  ZKX_VLOG_LINES(2, llvm_ir::DumpToString(module));

  return TranslateMLIRToLLVM(module, llvm_context);
}

void Postprocess(std::unique_ptr<llvm::Module> llvm_module,
                 llvm::Module* new_llvm_module,
                 KernelApiIrBuilder::KernelPrototype& kernel_prototype,
                 std::string_view name) {
  llvm::Linker linker(*new_llvm_module);
  bool failed = linker.linkInModule(std::move(llvm_module));
  CHECK(!failed) << "Linking failed";

  llvm::Function* impl_fn_c_iface =
      new_llvm_module->getFunction(absl::StrCat("_mlir_ciface_", name));
  CHECK(impl_fn_c_iface);

  llvm::Instruction* instr =
      kernel_prototype.function->getEntryBlock().getTerminator();
  llvm::IRBuilder<> b(instr);
  llvm::SmallVector<llvm::Value*> args;
  for (const auto& ir_array : kernel_prototype.arguments) {
    args.push_back(ir_array);
  }
  args.push_back(kernel_prototype.results[0]);
  b.CreateCall(impl_fn_c_iface, args);
}

}  // namespace

ElementalKernelEmitter::ElementalKernelEmitter(
    mlir::MLIRContext* context, const HloInstruction* instr,
    const BufferAssignment* buffer_assignment)
    : mlir_context_(context),
      instr_(instr),
      buffer_assignment_(buffer_assignment) {}

absl::StatusOr<KernelDefinition>
ElementalKernelEmitter::EmitKernelDefinition() {
  VLOG(2) << "Emit elemental host kernel: " << instr_->name();

  auto llvm_context = std::make_unique<llvm::LLVMContext>();

  const HloModule* hlo_module = instr_->GetModule();
  if (hlo_module == nullptr) {
    return absl::InternalError("HloModule is null");
  }

  LoadMlirDialects(mlir_context_);

  auto loc =
      mlir::NameLoc::get(mlir::StringAttr::get(mlir_context_, instr_->name()));
  EmitterLocOpBuilder b(loc, mlir_context_);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module(
      mlir::ModuleOp::create(std::move(loc)));
  b.setInsertionPointToEnd(mlir_module->getBody());

  llvm::SmallVector<mlir::Type> fn_arg_types;
  fn_arg_types.reserve(instr_->operand_count() + 1);
  for (int64_t i = 0; i < instr_->operand_count(); ++i) {
    const HloInstruction* operand = instr_->operand(i);
    fn_arg_types.push_back(
        llvm_ir::ShapeToMLIRMemRefType(operand->shape(), mlir_context_));
  }
  fn_arg_types.push_back(
      llvm_ir::ShapeToMLIRMemRefType(instr_->shape(), mlir_context_));

  auto fn = b.create<mlir::func::FuncOp>(
      instr_->name(), b.getFunctionType(fn_arg_types, std::nullopt));
  fn->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(mlir_context_));

  mlir::Block* entry_block = fn.addEntryBlock();
  b.setInsertionPointToEnd(entry_block);

  absl::flat_hash_map<const HloInstruction*, mlir::Value> values;
  for (int64_t i = 0; i < instr_->operand_count(); ++i) {
    const HloInstruction* operand = instr_->operand(i);
    if (ShapeUtil::IsScalar(operand->shape())) {
      values[operand] =
          b.create<mlir::memref::LoadOp>(entry_block->getArgument(i), {});
    } else {
      values[operand] = b.create<mlir::bufferization::ToTensorOp>(
          llvm_ir::ShapeToMLIRTensorType(operand->shape(), mlir_context_),
          entry_block->getArgument(i),
          /*restrict=*/true,
          /*writable=*/false);
    }
  }

  TF_ASSIGN_OR_RETURN(mlir::Value res, EmitOp(instr_, b, values));
  if (ShapeUtil::IsScalar(instr_->shape())) {
    b.create<mlir::memref::StoreOp>(
        res, entry_block->getArgument(instr_->operand_count()), {});
  } else {
    b.create<mlir::bufferization::MaterializeInDestinationOp>(
        mlir::TypeRange{}, res,
        entry_block->getArgument(instr_->operand_count()),
        /*restrict=*/true, /*writable=*/true);
  }

  b.create<mlir::func::ReturnOp>();

  std::unique_ptr<llvm::Module> llvm_module =
      CreateLLVMModule(mlir_context_, mlir_module.get(), llvm_context.get());

  KernelApiIrBuilder kernel_api_ir_builder(
      *llvm_context,
      KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()));

  std::unique_ptr<llvm::Module> new_llvm_module =
      KernelApiIrBuilder::CreateModule(
          absl::StrCat(instr_->name(), "_elemental_kernel_module"),
          *llvm_context);

  TF_ASSIGN_OR_RETURN(
      KernelApiIrBuilder::KernelPrototype kernel_prototype,
      kernel_api_ir_builder.EmitKernelPrototype(*new_llvm_module, instr_,
                                                buffer_assignment_, "_kernel"));

  Postprocess(std::move(llvm_module), new_llvm_module.get(), kernel_prototype,
              instr_->name());

  auto source = std::make_unique<LlvmIrKernelSource>(
      std::move(llvm_context), std::move(new_llvm_module));

  KernelSpec spec(kernel_prototype.function->getName(), se::ThreadDim(),
                  std::move(kernel_prototype.buffer_uses));

  return KernelDefinition(std::move(spec), std::move(source));
}

// static
absl::StatusOr<mlir::Value> ElementalKernelEmitter::EmitIntegerBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value, bool is_signed) {
  switch (instr->opcode()) {
    // TODO(jingyue): add the "nsw" attribute for signed types.
    case HloOpcode::kAdd:
      return b.create<mlir::arith::AddIOp>(lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return b.create<mlir::arith::MulIOp>(lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return b.create<mlir::arith::SubIOp>(lhs_value, rhs_value);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary integer op: %s", HloOpcodeString(instr->opcode())));
  }
}

// static
absl::StatusOr<mlir::Value> ElementalKernelEmitter::EmitFieldBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value) {
  switch (instr->opcode()) {
    case HloOpcode::kAdd:
      return b.create<mlir::zkir::field::AddOp>(lhs_value, rhs_value);
    case HloOpcode::kMultiply:
      return b.create<mlir::zkir::field::MulOp>(lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return b.create<mlir::zkir::field::SubOp>(lhs_value, rhs_value);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary field op: %s", HloOpcodeString(instr->opcode())));
  }
}

// static
absl::StatusOr<mlir::Value> ElementalKernelEmitter::EmitEcPointBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value) {
  mlir::Type ret_type = llvm_ir::PrimitiveTypeToMLIRType(
      instr->shape().element_type(), b.getContext());
  switch (instr->opcode()) {
    case HloOpcode::kAdd:
      return b.create<mlir::zkir::elliptic_curve::AddOp>(ret_type, lhs_value,
                                                         rhs_value);
      break;
    case HloOpcode::kMultiply:
      return b.create<mlir::zkir::elliptic_curve::ScalarMulOp>(
          ret_type, lhs_value, rhs_value);
      break;
    case HloOpcode::kSubtract:
      return b.create<mlir::zkir::elliptic_curve::SubOp>(ret_type, lhs_value,
                                                         rhs_value);
      break;
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary ec point op %s", HloOpcodeString(instr->opcode())));
  }
}

// static
absl::StatusOr<mlir::Value> ElementalKernelEmitter::EmitBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value) {
  Shape shape = instr->operand(0)->shape();
  PrimitiveType operand_type = shape.element_type();
  if (ShapeUtil::ElementIsIntegral(shape)) {
    return EmitIntegerBinaryOp(
        instr, b, lhs_value, rhs_value,
        primitive_util::IsSignedIntegralType(operand_type));
  } else if (ShapeUtil::ElementIsEcPoint(shape) ||
             ShapeUtil::ElementIsEcPoint(instr->operand(1)->shape())) {
    return EmitEcPointBinaryOp(instr, b, lhs_value, rhs_value);
  } else if (ShapeUtil::ElementIsField(shape)) {
    return EmitFieldBinaryOp(instr, b, lhs_value, rhs_value);
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Unhandled primitive type: %s",
      primitive_util::LowercasePrimitiveTypeName(operand_type)));
}

// static
absl::StatusOr<mlir::Value> ElementalKernelEmitter::EmitOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b,
    absl::flat_hash_map<const HloInstruction*, mlir::Value>& values) {
  switch (instr->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract: {
      return EmitBinaryOp(instr, b, values[instr->operand(0)],
                          values[instr->operand(1)]);
    }
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Unhandled opcode for elemental IR emission: %s",
                          HloOpcodeString(instr->opcode())));
  }
}

}  // namespace zkx::cpu
