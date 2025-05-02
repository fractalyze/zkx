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
#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "zkx/base/logging.h"
#include "zkx/codegen/emitter_loc_op_builder.h"
#include "zkx/codegen/llvm_ir_kernel_source.h"
#include "zkx/service/elemental_ir_emitter.h"
#include "zkx/service/llvm_ir/llvm_util.h"

namespace zkx::cpu {

namespace {

void LoadMlirDialects(mlir::MLIRContext* mlir_context) {
  mlir_context
      ->loadDialect<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();
}

void AddPasses(mlir::OpPassManager& pm) {
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertToLLVMPass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

std::unique_ptr<llvm::Module> TranslateMLIRToLLVM(
    mlir::ModuleOp module, llvm::LLVMContext* llvm_context) {
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(module, *llvm_context);
  CHECK(llvm_module) << "Failed to translate module to LLVM IR.";
  VLOG(1) << "After translation to LLVM: "
          << llvm_ir::DumpToString(llvm_module.get());
  return llvm_module;
}

std::unique_ptr<llvm::Module> CreateLLVMModule(
    mlir::MLIRContext* mlir_context, mlir::ModuleOp module,
    llvm::LLVMContext* llvm_context) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerAllExtensions(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  module->getContext()->appendDialectRegistry(registry);

  VLOG(1) << "Before optimization: " << llvm_ir::DumpToString(module);
  mlir::PassManager pm(mlir_context);
  AddPasses(pm);

  CHECK(mlir::succeeded(pm.run(module)));

  VLOG(1) << "After optimization: " << llvm_ir::DumpToString(module);

  return TranslateMLIRToLLVM(module, llvm_context);
}

void Postprocess(llvm::Module* llvm_module, llvm::Module* new_llvm_module,
                 KernelApiIrBuilder::KernelPrototype& kernel_prototype,
                 std::string_view name) {
  llvm::Function* impl_fn = llvm_module->getFunction(name);
  CHECK(impl_fn);

  llvm::Function* from_fn = impl_fn;
  llvm::Function* to_fn = kernel_prototype.function;

  llvm::BasicBlock* last_block = &to_fn->back();
  to_fn->splice(llvm::Function::iterator(last_block), from_fn);

  llvm::BranchInst* br = llvm::dyn_cast<llvm::BranchInst>(
      kernel_prototype.function->getEntryBlock().getTerminator());
  CHECK(br && br->isUnconditional());
  llvm::Function::iterator it = std::next(to_fn->begin());
  llvm::BasicBlock* target_block = &*it;
  br->setSuccessor(0, target_block);

  it = std::prev(std::prev(to_fn->end()));
  llvm::BasicBlock* from_block = &*it;
  from_block->getTerminator()->eraseFromParent();
  llvm::IRBuilder<> b(from_block);
  b.CreateBr(last_block);

  for (const auto& [arg, ir_array] :
       llvm::zip(impl_fn->args(), kernel_prototype.arguments)) {
    arg.replaceAllUsesWith(ir_array.GetBasePointer());
  }

  llvm::Argument* scratchpad_arg = impl_fn->getArg(impl_fn->arg_size() - 1);
  scratchpad_arg->replaceAllUsesWith(
      kernel_prototype.results[0].GetBasePointer());
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
    fn_arg_types.push_back(b.getType<mlir::LLVM::LLVMPointerType>());
  }
  fn_arg_types.push_back(b.getType<mlir::LLVM::LLVMPointerType>());

  auto fn = b.create<mlir::func::FuncOp>(
      instr_->name(), b.getFunctionType(fn_arg_types, std::nullopt));

  // The `entry_block` is used by `MlirForLoop` to allocate the induction
  // variable within this block.
  mlir::Block* entry_block = fn.addEntryBlock();
  mlir::Block* block = fn.addBlock();
  b.setInsertionPointToEnd(entry_block);
  b.create<mlir::cf::BranchOp>(block);
  b.setInsertionPointToStart(block);

  auto get_mlir_array = [this, entry_block](const HloInstruction* instr,
                                            int64_t i) {
    return llvm_ir::MlirArray(
        entry_block->getArgument(i),
        llvm_ir::ShapeToMLIRType(instr->shape(), mlir_context_),
        instr->shape());
  };

  std::vector<llvm_ir::MlirArray> arguments;
  arguments.reserve(instr_->operand_count());
  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (int64_t i = 0; i < instr_->operand_count(); ++i) {
    const HloInstruction* operand = instr_->operand(i);
    arguments.push_back(get_mlir_array(operand, i));
    operand_to_generator[operand] = [&arguments, &b,
                                     i](const llvm_ir::MlirArray::Index& idx) {
      return arguments[i].EmitReadArrayElement(idx, b);
    };
  }

  ElementalIrEmitter elemental_ir_emitter(b);

  llvm_ir::ElementGenerator element_generator =
      elemental_ir_emitter.MakeElementGenerator(instr_, operand_to_generator);

  llvm_ir::MlirArray result = get_mlir_array(instr_, instr_->operand_count());
  TF_ASSIGN_OR_RETURN(se::ThreadDim thread_dims,
                      EmitElementalLoops(b, instr_, result, element_generator));

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

  Postprocess(llvm_module.get(), new_llvm_module.get(), kernel_prototype,
              instr_->name());

  auto source = std::make_unique<LlvmIrKernelSource>(
      std::move(llvm_context), std::move(new_llvm_module));

  KernelSpec spec(kernel_prototype.function->getName(), thread_dims,
                  std::move(kernel_prototype.buffer_uses));

  return KernelDefinition(std::move(spec), std::move(source));
}

absl::StatusOr<se::ThreadDim> ElementalKernelEmitter::EmitElementalLoops(
    EmitterLocOpBuilder& b, const HloInstruction* instr,
    const llvm_ir::MlirArray& result,
    const llvm_ir::ElementGenerator& element_generator) {
  // clang-format off
  // TODO(chokboole): Implement parallel loop. Dependency: ShapePartitionAssigner.
  // clang-format on

  // Emit a whole loop for the instruction.
  TF_RETURN_IF_ERROR(llvm_ir::MlirLoopEmitter(element_generator, result, b)
                         .EmitLoop(llvm_ir::IrName(instr)));
  return se::ThreadDim();
}

}  // namespace zkx::cpu
