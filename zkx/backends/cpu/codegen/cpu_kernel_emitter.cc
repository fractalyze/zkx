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

#include "zkx/backends/cpu/codegen/cpu_kernel_emitter.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#ifdef ZKX_HAS_OPENMP
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#endif

#include "xla/tsl/platform/cpu_info.h"
#include "xla/tsl/platform/statusor.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"
#include "zkir/Dialect/EllipticCurve/Conversions/EllipticCurveToLLVM/EllipticCurveToLLVM.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/Field/Conversions/ExtFieldToLLVM/ExtFieldToLLVM.h"
#include "zkir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"
#include "zkir/Dialect/Field/IR/FieldDialect.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/ModArith/Conversions/ModArithToArith/ModArithToArith.h"
#include "zkir/Dialect/ModArith/IR/ModArithDialect.h"
#include "zkir/Dialect/Poly/Conversions/PolyToField/PolyToField.h"
#include "zkir/Dialect/Poly/IR/PolyDialect.h"
#include "zkir/Dialect/Poly/IR/PolyOps.h"
#include "zkir/Dialect/TensorExt/Conversions/TensorExtToTensor/TensorExtToTensor.h"
#include "zkx/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "zkx/base/bits.h"
#include "zkx/base/logging.h"
#include "zkx/codegen/llvm_ir_kernel_source.h"
#include "zkx/layout_util.h"
#include "zkx/math/poly/root_of_unity.h"
#include "zkx/mlir/mlir_utils.h"
#include "zkx/primitive_util.h"
#include "zkx/service/llvm_ir/llvm_util.h"
#include "zkx/shape_util.h"

namespace zkx::cpu {

namespace {

template <typename T>
absl::StatusOr<mlir::zkir::field::RootOfUnityAttr> GetRootOfUnityAttr(
    mlir::MLIRContext* mlir_context, int64_t fft_length) {
  TF_ASSIGN_OR_RETURN(T root_of_unity, math::GetRootOfUnity<T>(fft_length));
  return mlir::zkir::field::RootOfUnityAttr::get(
      mlir_context, /*root=*/
      mlir_utils::GetMlirPrimeFieldAttr(mlir_context, root_of_unity,
                                        /*use_montgomery=*/false),
      /*degree=*/
      mlir::IntegerAttr::get(mlir::IntegerType::get(mlir_context, 64),
                             fft_length));
}

absl::StatusOr<mlir::Value> CreateZeroPoint(EmitterLocOpBuilder& b,
                                            const Shape& shape) {
  switch (shape.element_type()) {
#define MONTABLE_CASE(enum, cpp_type)                                  \
  case enum:                                                           \
    return mlir_utils::CreateMlirEcPointConstant(b, cpp_type::Zero()); \
  case enum##_STD:                                                     \
    return mlir_utils::CreateMlirEcPointConstant(b, cpp_type##Std::Zero());
    MONTABLE_CASE(BN254_G1_AFFINE, math::bn254::G1AffinePoint)
    MONTABLE_CASE(BN254_G1_JACOBIAN, math::bn254::G1JacobianPoint)
    MONTABLE_CASE(BN254_G1_XYZZ, math::bn254::G1PointXyzz)
    MONTABLE_CASE(BN254_G2_AFFINE, math::bn254::G2AffinePoint)
    MONTABLE_CASE(BN254_G2_JACOBIAN, math::bn254::G2JacobianPoint)
    MONTABLE_CASE(BN254_G2_XYZZ, math::bn254::G2PointXyzz)
#undef MONTABLE_CASE
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid primitive type: %s",
          primitive_util::LowercasePrimitiveTypeName(shape.element_type())));
  }
}

void LoadMlirDialects(mlir::MLIRContext* mlir_context) {
  mlir_context->loadDialect<
      // clang-format off
      mlir::arith::ArithDialect,
      mlir::bufferization::BufferizationDialect,
      mlir::cf::ControlFlowDialect,
      mlir::func::FuncDialect,
      mlir::linalg::LinalgDialect,
      mlir::LLVM::LLVMDialect,
      mlir::math::MathDialect,
      mlir::memref::MemRefDialect,
      mlir::scf::SCFDialect,
      mlir::sparse_tensor::SparseTensorDialect,
      mlir::zkir::elliptic_curve::EllipticCurveDialect,
      mlir::zkir::field::FieldDialect,
      mlir::zkir::mod_arith::ModArithDialect,
      mlir::zkir::poly::PolyDialect
      // clang-format on
      >();
}

void OneShotBufferize(mlir::OpPassManager& pm) {
  // NOTE: One-shot bufferize does not deallocate buffers. This is done by the
  // ownership-based buffer deallocation pass.
  // https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/
  mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  pm.addPass(mlir::memref::createExpandReallocPass());
  pm.addPass(mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(mlir::bufferization::createLowerDeallocationsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertBufferizationToMemRefPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void AddPasses(mlir::PassManager& pm, CpuKernelEmitter::PassFlag& flag) {
  if (VLOG_IS_ON(4)) {
    pm.getContext()->disableMultithreading();
    auto print_before = [](mlir::Pass*, mlir::Operation*) { return true; };
    auto print_after = [](mlir::Pass*, mlir::Operation*) { return true; };
    pm.enableIRPrinting(print_before, print_after, /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/true);
  }
  if (flag.enable_sparsification_and_bufferization) {
    VLOG(2) << "add pass: -sparsification-and-bufferization";
    flag.enable_expand_strided_metadata = true;
    mlir::bufferization::OneShotBufferizationOptions bufferization_options;
    bufferization_options.bufferizeFunctionBoundaries = true;
    mlir::SparsificationOptions sparsification_options;
    pm.addPass(mlir::createSparsificationAndBufferizationPass(
        bufferization_options, sparsification_options,
        /*createSparseDeallocs=*/false,
        /*enableRuntimeLibrary=*/false,
        /*enableBufferInitialization=*/false,
        /*vectorLength=*/0,
        /*enableVLAVectorization=*/false,
        /*enableSIMDIndex32=*/false,
        /*enableGPULibgen=*/false,
        /*sparseEmitStrategy=*/mlir::SparseEmitStrategy::kFunctional,
        /*parallelizationStrategy=*/
        mlir::SparseParallelizationStrategy::kDenseOuterLoop));
  }

  auto maybe_add_elementwise_to_linalg =
      [](mlir::PassManager& pm, CpuKernelEmitter::PassFlag& flag) -> void {
    if (flag.enable_elementwise_to_linalg) {
      VLOG(2) << "add pass: -convert-elementwise-to-linalg "
                 "-linalg-fuse-elementwise-ops";
      flag.enable_linalg_to_parallel_loops = true;
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::createConvertElementwiseToLinalgPass());
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::createLinalgElementwiseOpFusionPass());
    }
  };

  if (flag.enable_poly_to_field) {
    VLOG(2) << "add pass: -poly-to-field";
    flag.enable_field_to_arith = true;
    pm.addPass(mlir::zkir::poly::createPolyToField());
  }

  maybe_add_elementwise_to_linalg(pm, flag);

  if (flag.enable_elliptic_curve_to_field) {
    VLOG(2) << "add pass: -elliptic-curve-to-field";
    flag.enable_field_to_arith = true;
    pm.addPass(mlir::zkir::elliptic_curve::createEllipticCurveToField());
  }

  maybe_add_elementwise_to_linalg(pm, flag);

  if (flag.enable_field_to_arith) {
    VLOG(2) << "add pass: -field-to-mod-arith -mod-arith-to-arith";
    pm.addPass(mlir::zkir::field::createFieldToModArith());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::zkir::mod_arith::createModArithToArith());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  if (flag.enable_lower_affine) {
    VLOG(2) << "add pass: -lower-affine";
    flag.enable_scf_to_cf = true;
    pm.addPass(mlir::createLowerAffinePass());
  }

  if (flag.enable_tensor_ext_to_tensor) {
    VLOG(2) << "add pass: -tensor-ext-to-tensor";
    pm.addPass(mlir::zkir::tensor_ext::createTensorExtToTensor());
  }

  if (flag.enable_one_shot_bufferize) {
    VLOG(2) << "add pass: -one-shot-bufferize";
    OneShotBufferize(pm);
  }
  if (flag.enable_buffer_results_to_out_params) {
    VLOG(2) << "add pass: -buffer-results-to-out-params=hoist-static-allocs";
    mlir::bufferization::BufferResultsToOutParamsPassOptions out_params_options;
    out_params_options.hoistStaticAllocs = true;
    pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass(
        out_params_options));
  }

  if (flag.enable_linalg_to_parallel_loops) {
    VLOG(2) << "add pass: -convert-linalg-to-parallel-loops";
    flag.enable_scf_to_cf = true;
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::createConvertLinalgToParallelLoopsPass());
  }

  if (flag.enable_omp) {
#ifdef ZKX_HAS_OPENMP
    VLOG(2) << "add pass: -convert-scf-to-openmp";
    // NOTE(batzor): This pass introduces memref.alloca ops that have to be
    // lowered before the -convert-scf-to-cf pass.
    flag.enable_finalize_memref_to_llvm = true;
    pm.addPass(mlir::createConvertSCFToOpenMPPass());
#else
    VLOG(2) << "ZKX is not built with OpenMP. Skipping OpenMP pass...";
#endif
  }

  if (flag.enable_expand_strided_metadata) {
    VLOG(2) << "add pass: -expand-strided-metadata";
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::memref::createExpandStridedMetadataPass());

    VLOG(2) << "add pass: -lower-affine";
    flag.enable_scf_to_cf = true;
    pm.addPass(mlir::createLowerAffinePass());
  }

  if (flag.enable_finalize_memref_to_llvm) {
    VLOG(2) << "add pass: -finalize-memref-to-llvm";
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  }

  if (flag.enable_scf_to_cf) {
    VLOG(2) << "add pass: -convert-scf-to-cf";
    pm.addPass(mlir::createSCFToControlFlowPass());
  }
  if (flag.enable_elliptic_curve_to_llvm) {
    VLOG(2) << "add pass: -convert-ec-to-llvm";
    pm.addPass(mlir::zkir::elliptic_curve::createEllipticCurveToLLVM());
  }
  if (flag.enable_ext_field_to_llvm) {
    VLOG(2) << "add pass: -convert-ext-field-to-llvm";
    pm.addPass(mlir::zkir::field::createExtFieldToLLVM());
  }
  pm.addPass(mlir::createConvertToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
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
    llvm::LLVMContext* llvm_context, CpuKernelEmitter::PassFlag& pass_flag) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
#ifdef ZKX_HAS_OPENMP
  mlir::registerOpenMPDialectTranslation(registry);
#endif
  mlir::registerAllExtensions(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  if (pass_flag.enable_one_shot_bufferize) {
    mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::sparse_tensor::registerBufferizableOpInterfaceExternalModels(
        registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  }
  if (pass_flag.enable_elliptic_curve_to_llvm) {
    mlir::zkir::elliptic_curve::registerConvertEllipticCurveToLLVMInterface(
        registry);
  }
  if (pass_flag.enable_ext_field_to_llvm) {
    mlir::zkir::field::registerConvertExtFieldToLLVMInterface(registry);
  }
  module->getContext()->appendDialectRegistry(registry);

  VLOG(2) << "MLIR before optimizations";
  ZKX_VLOG_LINES(2, llvm_ir::DumpToString(module));
  mlir::PassManager pm(mlir_context);
  AddPasses(pm, pass_flag);

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

CpuKernelEmitter::CpuKernelEmitter(mlir::MLIRContext* context,
                                   const HloInstruction* instr,
                                   const BufferAssignment* buffer_assignment)
    : mlir_context_(context),
      instr_(instr),
      buffer_assignment_(buffer_assignment) {}

absl::StatusOr<llvm::SmallVector<mlir::Type>>
CpuKernelEmitter::MakeFuncArguments() const {
  llvm::SmallVector<mlir::Type> args;
  args.reserve(instr_->operand_count() + 1);
  for (int64_t i = 0; i < instr_->operand_count(); ++i) {
    const HloInstruction* operand = instr_->operand(i);
    const Shape& shape = operand->shape();
    if (LayoutUtil::IsSparseArray(shape)) {
      args.push_back(mlir_utils::ShapeToMlirMemRefType(
          ShapeUtil::MakeShape(U8, {ShapeUtil::SparseArrayDataSize(shape)}),
          mlir_context_));
    } else {
      args.push_back(mlir_utils::ShapeToMlirMemRefType(shape, mlir_context_));
    }
  }
  return std::move(args);
}

absl::StatusOr<llvm::SmallVector<mlir::Type>>
CpuKernelEmitter::MakeFuncReturnTypes() const {
  if (LayoutUtil::IsSparseArray(instr_->shape())) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unhandled sparse array layout: %s", instr_->ToString()));
  }
  llvm::SmallVector<mlir::Type> ret;
  ret.push_back(
      mlir_utils::ShapeToMlirMemRefType(instr_->shape(), mlir_context_));
  return std::move(ret);
}

mlir::Value CpuKernelEmitter::EmitCSROperand(EmitterLocOpBuilder& b,
                                             mlir::Block* entry_block,
                                             int64_t i,
                                             const Shape& shape) const {
  int64_t num_rows = shape.dimensions(0);
  int64_t num_nonzeros = shape.layout().num_nonzeros();

  int64_t offset0 = 0;
  int64_t offset1 = offset0 + (num_rows + 1) * sizeof(uint32_t);
  int64_t offset2 = offset1 + num_nonzeros * sizeof(uint32_t);

  auto offset0_op = b.create<mlir::arith::ConstantIndexOp>(offset0);
  auto offset1_op = b.create<mlir::arith::ConstantIndexOp>(offset1);
  auto offset2_op = b.create<mlir::arith::ConstantIndexOp>(offset2);

  Shape row_ptrs_shape = ShapeUtil::MakeShape(U32, {num_rows + 1});
  Shape col_indices_shape = ShapeUtil::MakeShape(U32, {num_nonzeros});
  Shape values_array_shape =
      ShapeUtil::MakeShape(shape.element_type(), {num_nonzeros});

  auto row_ptrs_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(row_ptrs_shape, mlir_context_),
      entry_block->getArgument(i), offset0_op, mlir::ValueRange{});
  auto col_indices_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(col_indices_shape, mlir_context_),
      entry_block->getArgument(i), offset1_op, mlir::ValueRange{});
  auto values_array_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(values_array_shape, mlir_context_),
      entry_block->getArgument(i), offset2_op, mlir::ValueRange{});

  auto row_ptrs = b.create<mlir::bufferization::ToTensorOp>(
      mlir_utils::ShapeToMlirTensorType(row_ptrs_shape, mlir_context_),
      row_ptrs_memref,
      /*restrict=*/true,
      /*writable=*/false);
  auto col_indices = b.create<mlir::bufferization::ToTensorOp>(
      mlir_utils::ShapeToMlirTensorType(col_indices_shape, mlir_context_),
      col_indices_memref,
      /*restrict=*/true,
      /*writable=*/false);
  auto values_array = b.create<mlir::bufferization::ToTensorOp>(
      mlir_utils::ShapeToMlirTensorType(values_array_shape, mlir_context_),
      values_array_memref,
      /*restrict=*/true,
      /*writable=*/false);

  return b.create<mlir::sparse_tensor::AssembleOp>(
      mlir_utils::ShapeToMlirTensorType(shape, mlir_context_),
      mlir::ValueRange{row_ptrs, col_indices}, values_array);
}

mlir::Value CpuKernelEmitter::EmitCSCOperand(EmitterLocOpBuilder& b,
                                             mlir::Block* entry_block,
                                             int64_t i,
                                             const Shape& shape) const {
  int64_t num_cols = shape.dimensions(0);
  int64_t num_nonzeros = shape.layout().num_nonzeros();

  int64_t offset0 = 0;
  int64_t offset1 = offset0 + (num_cols + 1) * sizeof(uint32_t);
  int64_t offset2 = offset1 + num_nonzeros * sizeof(uint32_t);

  auto offset0_op = b.create<mlir::arith::ConstantIndexOp>(offset0);
  auto offset1_op = b.create<mlir::arith::ConstantIndexOp>(offset1);
  auto offset2_op = b.create<mlir::arith::ConstantIndexOp>(offset2);

  Shape col_ptrs_shape = ShapeUtil::MakeShape(U32, {num_cols + 1});
  Shape row_indices_shape = ShapeUtil::MakeShape(U32, {num_nonzeros});
  Shape values_array_shape =
      ShapeUtil::MakeShape(shape.element_type(), {num_nonzeros});

  auto col_ptrs_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(col_ptrs_shape, mlir_context_),
      entry_block->getArgument(i), offset0_op, mlir::ValueRange{});
  auto row_indices_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(row_indices_shape, mlir_context_),
      entry_block->getArgument(i), offset1_op, mlir::ValueRange{});
  auto values_array_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(values_array_shape, mlir_context_),
      entry_block->getArgument(i), offset2_op, mlir::ValueRange{});

  auto col_ptrs = b.create<mlir::bufferization::ToTensorOp>(
      mlir_utils::ShapeToMlirTensorType(col_ptrs_shape, mlir_context_),
      col_ptrs_memref,
      /*restrict=*/true,
      /*writable=*/false);
  auto row_indices = b.create<mlir::bufferization::ToTensorOp>(
      mlir_utils::ShapeToMlirTensorType(row_indices_shape, mlir_context_),
      row_indices_memref,
      /*restrict=*/true,
      /*writable=*/false);
  auto values_array = b.create<mlir::bufferization::ToTensorOp>(
      mlir_utils::ShapeToMlirTensorType(values_array_shape, mlir_context_),
      values_array_memref,
      /*restrict=*/true,
      /*writable=*/false);

  return b.create<mlir::sparse_tensor::AssembleOp>(
      mlir_utils::ShapeToMlirTensorType(shape, mlir_context_),
      mlir::ValueRange{col_ptrs, row_indices}, values_array);
}

mlir::Value CpuKernelEmitter::EmitCOOOperand(EmitterLocOpBuilder& b,
                                             mlir::Block* entry_block,
                                             int64_t i,
                                             const Shape& shape) const {
  int64_t num_nonzeros = shape.layout().num_nonzeros();
  CHECK_LE(num_nonzeros, std::numeric_limits<int32_t>::max());

  auto positions_type = mlir::RankedTensorType::get({2}, b.getI32Type());
  auto positions_attr = mlir::DenseElementsAttr::get(
      positions_type,
      llvm::ArrayRef<int32_t>{0, static_cast<int32_t>(num_nonzeros)});
  auto positions =
      b.create<mlir::arith::ConstantOp>(positions_type, positions_attr);

  int64_t offset0 = 0;
  int64_t offset1 = offset0 + num_nonzeros * sizeof(uint32_t) * 2;

  auto offset0_op = b.create<mlir::arith::ConstantIndexOp>(offset0);
  auto offset1_op = b.create<mlir::arith::ConstantIndexOp>(offset1);

  Shape indices_shape = ShapeUtil::MakeShape(U32, {num_nonzeros, 2});
  Shape values_array_shape =
      ShapeUtil::MakeShape(shape.element_type(), {num_nonzeros});

  auto indices_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(indices_shape, mlir_context_),
      entry_block->getArgument(i), offset0_op, mlir::ValueRange{});
  auto values_array_memref = b.create<mlir::memref::ViewOp>(
      mlir_utils::ShapeToMlirMemRefType(values_array_shape, mlir_context_),
      entry_block->getArgument(i), offset1_op, mlir::ValueRange{});

  auto indices = b.create<mlir::bufferization::ToTensorOp>(
      mlir_utils::ShapeToMlirTensorType(indices_shape, mlir_context_),
      indices_memref,
      /*restrict=*/true,
      /*writable=*/false);
  auto values_array = b.create<mlir::bufferization::ToTensorOp>(
      mlir_utils::ShapeToMlirTensorType(values_array_shape, mlir_context_),
      values_array_memref,
      /*restrict=*/true,
      /*writable=*/false);

  return b.create<mlir::sparse_tensor::AssembleOp>(
      mlir_utils::ShapeToMlirTensorType(shape, mlir_context_),
      mlir::ValueRange{positions, indices}, values_array);
}

absl::StatusOr<absl::flat_hash_map<const HloInstruction*, mlir::Value>>
CpuKernelEmitter::EmitOperands(EmitterLocOpBuilder& b,
                               mlir::Block* entry_block) const {
  absl::flat_hash_map<const HloInstruction*, mlir::Value> values;
  for (int64_t i = 0; i < instr_->operand_count(); ++i) {
    const HloInstruction* operand = instr_->operand(i);
    const Shape& shape = operand->shape();
    if (LayoutUtil::IsSparseArray(shape)) {
      pass_flag_.enable_sparsification_and_bufferization = true;

      if (LayoutUtil::IsCSRArray(shape)) {
        values[operand] = EmitCSROperand(b, entry_block, i, shape);
      } else if (LayoutUtil::IsCSCArray(shape)) {
        values[operand] = EmitCSCOperand(b, entry_block, i, shape);
      } else if (LayoutUtil::IsCOOArray(shape)) {
        values[operand] = EmitCOOOperand(b, entry_block, i, shape);
      } else {
        return absl::UnimplementedError(absl::StrFormat(
            "Unhandled sparse array layout: %s", operand->ToString()));
      }
    } else if (ShapeUtil::IsScalar(shape)) {
      values[operand] =
          b.create<mlir::memref::LoadOp>(entry_block->getArgument(i), {});
    } else {
      pass_flag_.enable_one_shot_bufferize = true;

      values[operand] = b.create<mlir::bufferization::ToTensorOp>(
          mlir_utils::ShapeToMlirTensorType(shape, mlir_context_),
          entry_block->getArgument(i),
          /*restrict=*/true,
          /*writable=*/false);
    }
  }
  return std::move(values);
}

absl::Status CpuKernelEmitter::EmitEpilog(EmitterLocOpBuilder& b,
                                          mlir::Block* entry_block,
                                          mlir::MemRefType ret_type,
                                          mlir::Value result) const {
  const Shape& shape = instr_->shape();
  mlir::Value ret_value;
  if (LayoutUtil::IsSparseArray(shape)) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unhandled sparse array layout: %s", instr_->ToString()));
  } else if (ShapeUtil::IsScalar(shape)) {
    ret_value = b.create<mlir::memref::AllocOp>(ret_type);
    b.create<mlir::memref::StoreOp>(result, ret_value, {});
  } else {
    pass_flag_.enable_one_shot_bufferize = true;

    ret_value = b.create<mlir::bufferization::ToBufferOp>(ret_type, result);
  }

  b.create<mlir::func::ReturnOp>(mlir::ValueRange{ret_value});
  return absl::OkStatus();
}

absl::StatusOr<KernelDefinition> CpuKernelEmitter::EmitKernelDefinition() {
  VLOG(2) << "Emit host kernel: " << instr_->name();

  auto llvm_context = std::make_unique<llvm::LLVMContext>();

  const HloModule* hlo_module = instr_->GetModule();
  if (hlo_module == nullptr) {
    return absl::InternalError("HloModule is null");
  }

  LoadMlirDialects(mlir_context_);

  auto loc =
      mlir::NameLoc::get(mlir::StringAttr::get(mlir_context_, instr_->name()));
  EmitterLocOpBuilder b(loc, mlir_context_);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      llvm_ir::CreateMlirModuleOp(std::move(loc));
  b.setInsertionPointToEnd(mlir_module->getBody());

  TF_ASSIGN_OR_RETURN(llvm::SmallVector<mlir::Type> fn_arg_types,
                      MakeFuncArguments());
  llvm::SmallVector<mlir::Type> fn_ret_types;
  mlir::func::FuncOp fn;
  TF_ASSIGN_OR_RETURN(fn_ret_types, MakeFuncReturnTypes());
  fn = b.create<mlir::func::FuncOp>(
      instr_->name(), b.getFunctionType(fn_arg_types, fn_ret_types));
  for (int64_t i = 0; i < fn_arg_types.size(); ++i) {
    fn.setArgAttr(i,
                  mlir::bufferization::BufferizationDialect::kWritableAttrName,
                  b.getBoolAttr(false));
  }
  fn->setAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
              mlir::UnitAttr::get(mlir_context_));

  mlir::Block* entry_block = fn.addEntryBlock();
  b.setInsertionPointToEnd(entry_block);

  absl::flat_hash_map<const HloInstruction*, mlir::Value> values;
  TF_ASSIGN_OR_RETURN(values, EmitOperands(b, entry_block));

  TF_ASSIGN_OR_RETURN(mlir::Value res, EmitOp(instr_, b, values));

  TF_RETURN_IF_ERROR(EmitEpilog(
      b, entry_block, mlir::cast<mlir::MemRefType>(fn_ret_types[0]), res));

  std::unique_ptr<llvm::Module> llvm_module = CreateLLVMModule(
      mlir_context_, mlir_module.get(), llvm_context.get(), pass_flag_);

  KernelApiIrBuilder kernel_api_ir_builder(
      *llvm_context,
      KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()));

  std::unique_ptr<llvm::Module> new_llvm_module =
      KernelApiIrBuilder::CreateModule(
          absl::StrCat(instr_->name(), "_kernel_module"), *llvm_context);

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

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitIntegerUnaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value) {
  switch (instr->opcode()) {
    case HloOpcode::kNegate:
      return b.create<mlir::arith::SubIOp>(
          b.create<mlir::arith::ConstantOp>(b.getZeroAttr(value.getType())),
          value);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled unary integer op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitFieldUnaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value) {
  switch (instr->opcode()) {
    case HloOpcode::kConvert: {
      PrimitiveType from = instr->operand(0)->shape().element_type();
      PrimitiveType to = instr->shape().element_type();
      if (primitive_util::IsMontgomeryForm(from) &&
          primitive_util::IsMontgomeryForm(to)) {
        return value;
      } else if (primitive_util::IsMontgomeryForm(from) &&
                 !primitive_util::IsMontgomeryForm(to)) {
        return b.create<mlir::zkir::field::FromMontOp>(
            mlir::zkir::field::getStandardFormType(value.getType()), value);
      } else if (!primitive_util::IsMontgomeryForm(from) &&
                 primitive_util::IsMontgomeryForm(to)) {
        return b.create<mlir::zkir::field::ToMontOp>(
            mlir::zkir::field::getMontgomeryFormType(value.getType()), value);
      } else {
        return value;
      }
    }
    case HloOpcode::kInverse:
      return b.create<mlir::zkir::field::InverseOp>(value);
    case HloOpcode::kNegate:
      return b.create<mlir::zkir::field::NegateOp>(value);

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled unary field op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitEcPointUnaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value) {
  switch (instr->opcode()) {
    case HloOpcode::kConvert: {
      const PrimitiveType from = instr->operand(0)->shape().element_type();
      const PrimitiveType to = instr->shape().element_type();
      if (from == to) {
        return value;
      }
      return b.create<mlir::zkir::elliptic_curve::ConvertPointTypeOp>(
          mlir_utils::PrimitiveTypeToMlirType(to, b.getContext()), value);
    }
    case HloOpcode::kNegate:
      return b.create<mlir::zkir::elliptic_curve::NegateOp>(value);

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled unary ec point op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitUnaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value) {
  Shape shape = instr->operand(0)->shape();
  PrimitiveType operand_type = shape.element_type();
  if (ShapeUtil::ElementIsIntegral(shape)) {
    return EmitIntegerUnaryOp(instr, b, value);
  } else if (ShapeUtil::ElementIsField(shape)) {
    return EmitFieldUnaryOp(instr, b, value);
  } else if (ShapeUtil::ElementIsEcPoint(shape)) {
    return EmitEcPointUnaryOp(instr, b, value);
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Unhandled primitive type: %s",
      primitive_util::LowercasePrimitiveTypeName(operand_type)));
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitIntegerBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value, bool is_signed) {
  switch (instr->opcode()) {
    // TODO(jingyue): add the "nsw" attribute for signed types.
    case HloOpcode::kAdd:
      return b.create<mlir::arith::AddIOp>(lhs_value, rhs_value);
    case HloOpcode::kDivide: {
      if (is_signed) {
        return b.create<mlir::arith::DivSIOp>(lhs_value, rhs_value);
      } else {
        return b.create<mlir::arith::DivUIOp>(lhs_value, rhs_value);
      }
    }
    case HloOpcode::kMultiply:
      return b.create<mlir::arith::MulIOp>(lhs_value, rhs_value);
    case HloOpcode::kPower:
      return b.create<mlir::math::IPowIOp>(lhs_value, rhs_value);
    case HloOpcode::kSubtract:
      return b.create<mlir::arith::SubIOp>(lhs_value, rhs_value);

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary integer op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitFieldBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value) {
  switch (instr->opcode()) {
    case HloOpcode::kAdd:
      return b.create<mlir::zkir::field::AddOp>(lhs_value, rhs_value);
    case HloOpcode::kDivide: {
      auto inv = b.create<mlir::zkir::field::InverseOp>(rhs_value);
      return b.create<mlir::zkir::field::MulOp>(lhs_value, inv);
    }
    case HloOpcode::kMultiply:
      return b.create<mlir::zkir::field::MulOp>(lhs_value, rhs_value);
    case HloOpcode::kPower: {
      const PrimitiveType exponent_type =
          instr->operand(1)->shape().element_type();
      if (!primitive_util::IsUnsignedIntegralType(exponent_type)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "The exponent for a power operation on a field must be an "
            "unsigned integer, but got %s",
            primitive_util::LowercasePrimitiveTypeName(exponent_type)));
      }
      return b.create<mlir::zkir::field::PowUIOp>(lhs_value.getType(),
                                                  lhs_value, rhs_value);
    }
    case HloOpcode::kSubtract:
      return b.create<mlir::zkir::field::SubOp>(lhs_value, rhs_value);
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled binary field op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitEcPointBinaryOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs_value,
    mlir::Value rhs_value) {
  const Shape& shape = instr->shape();
  mlir::Type ret_type;
  if (ShapeUtil::IsScalar(shape)) {
    ret_type = mlir_utils::PrimitiveTypeToMlirType(shape.element_type(),
                                                   b.getContext());
  } else {
    ret_type = mlir_utils::ShapeToMlirTensorType(shape, b.getContext());
  }

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

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitBinaryOp(
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

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitFftOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value,
    mlir::Value twiddle_factor) {
  pass_flag_.enable_poly_to_field = true;
  pass_flag_.enable_lower_affine = true;
  if (instr->fft_type() == FftType::IFFT) {
    pass_flag_.enable_linalg_to_parallel_loops = true;
  }

  PrimitiveType operand_type = instr->operand(0)->shape().element_type();
  absl::StatusOr<mlir::zkir::field::RootOfUnityAttr> root;
  mlir::zkir::field::RootOfUnityAttr root_attr;
  switch (operand_type) {
    case BN254_SCALAR: {
      root = GetRootOfUnityAttr<math::bn254::Fr>(value.getContext(),
                                                 instr->fft_length());
      break;
    }
    case BN254_SCALAR_STD: {
      root = GetRootOfUnityAttr<math::bn254::FrStd>(value.getContext(),
                                                    instr->fft_length());
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid primitive type: %s",
          primitive_util::LowercasePrimitiveTypeName(operand_type)));
  }
  if (!root.ok()) return root.status();
  root_attr = root.value();
  pass_flag_.enable_tensor_ext_to_tensor = true;

  auto alloc_tensor = b.create<mlir::bufferization::AllocTensorOp>(
      mlir_utils::ShapeToMlirTensorType(instr->shape(), b.getContext()),
      mlir::ValueRange{});
  switch (instr->fft_type()) {
    case FftType::FFT: {
      return b.create<mlir::zkir::poly::NTTOp>(
          value, alloc_tensor, twiddle_factor, root_attr,
          /*tileX=*/nullptr, /*gridSizeX=*/nullptr,
          instr->fft_do_bit_reverse());
    }
    case FftType::IFFT: {
      return b.create<mlir::zkir::poly::NTTOp>(
          value, alloc_tensor, twiddle_factor, root_attr,
          /*tileX=*/nullptr, /*gridSizeX=*/nullptr, instr->fft_do_bit_reverse(),
          /*inverse=*/true);
    }

    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled fft type: %s", FftType_Name(instr->fft_type())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitMsmOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value scalars,
    mlir::Value bases) {
  pass_flag_.enable_elementwise_to_linalg = true;

  if (auto tensor_type =
          mlir::dyn_cast<mlir::RankedTensorType>(bases.getType())) {
    int64_t num_scalar_mul = instr->operand(0)->shape().dimensions(0);
    CHECK_GT(num_scalar_mul, 0);
    int64_t num_threads = tsl::port::MaxParallelism();
    int32_t window_bits = instr->window_bits();

    const Shape& shape = instr->shape();
    auto result_type = mlir_utils::PrimitiveTypeToMlirType(shape.element_type(),
                                                           b.getContext());
    if (num_scalar_mul < num_threads) {
      int32_t degree = static_cast<int32_t>(
          base::Log2Ceiling(static_cast<uint64_t>(num_scalar_mul)));
      return b.create<mlir::zkir::elliptic_curve::MSMOp>(
          result_type, scalars, bases, degree, window_bits);
    }

    pass_flag_.enable_expand_strided_metadata = true;

    int64_t chunk_size = (num_scalar_mul + num_threads - 1) / num_threads;
    int32_t degree = static_cast<int32_t>(
        base::Log2Ceiling(static_cast<uint64_t>(chunk_size)));

    auto zero = b.create<mlir::arith::ConstantIndexOp>(0);
    auto one = b.create<mlir::arith::ConstantIndexOp>(1);

    mlir::Value num_scalar_mul_value =
        b.create<mlir::arith::ConstantIndexOp>(num_scalar_mul);
    mlir::Value num_threads_value =
        b.create<mlir::arith::ConstantIndexOp>(num_threads);
    mlir::Value chunk_size_value =
        b.create<mlir::arith::ConstantIndexOp>(chunk_size);

    TF_ASSIGN_OR_RETURN(mlir::Value zero_point, CreateZeroPoint(b, shape));

    auto results_type = mlir::MemRefType::get({num_threads}, result_type);
    mlir::Value results = b.create<mlir::memref::AllocOp>(results_type);

    // Initialize results
    b.create<mlir::scf::ForOp>(
        /*lowerBound=*/zero, /*upperBound=*/num_threads_value, /*step=*/one,
        /*initArgs=*/std::nullopt,
        [zero_point, results](mlir::OpBuilder& nested_b, mlir::Location loc,
                              mlir::Value iv, mlir::ValueRange loop_vars) {
          mlir::ImplicitLocOpBuilder b(loc, nested_b);
          b.create<mlir::memref::StoreOp>(zero_point, results, iv);
          b.create<mlir::scf::YieldOp>();
        });

    // Parallel MSM computation
    b.create<mlir::scf::ParallelOp>(
        /*lowerBounds=*/mlir::ValueRange{zero},
        /*upperBounds=*/mlir::ValueRange{num_threads_value},
        /*steps=*/mlir::ValueRange{one},
        [&](mlir::OpBuilder& nested_b, mlir::Location loc,
            mlir::ValueRange ivs) {
          mlir::ImplicitLocOpBuilder b(loc, nested_b);
          auto i = ivs[0];

          auto start = b.create<mlir::arith::MulIOp>(i, chunk_size_value);
          auto next_i = b.create<mlir::arith::AddIOp>(i, one);
          auto end = b.create<mlir::arith::MulIOp>(next_i, chunk_size_value);

          auto actual_end =
              b.create<mlir::arith::MinUIOp>(end, num_scalar_mul_value);
          auto actual_chunk_size =
              b.create<mlir::arith::SubIOp>(actual_end, start);

          // Extract chunks
          auto scalars_chunk = b.create<mlir::tensor::ExtractSliceOp>(
              scalars, mlir::ValueRange{start},
              mlir::ValueRange{actual_chunk_size}, mlir::ValueRange{one});
          auto bases_chunk = b.create<mlir::tensor::ExtractSliceOp>(
              bases, mlir::ValueRange{start},
              mlir::ValueRange{actual_chunk_size}, mlir::ValueRange{one});

          // Compute MSM for chunk
          auto msm_result = b.create<mlir::zkir::elliptic_curve::MSMOp>(
              result_type, scalars_chunk, bases_chunk, degree, window_bits);

          b.create<mlir::memref::StoreOp>(msm_result, results, i);
        });

    // Combine results using reduction loop
    auto final_result = b.create<mlir::scf::ForOp>(
        /*lowerBound=*/zero, /*upperBound=*/num_threads_value, /*step=*/one,
        /*initArgs=*/mlir::ValueRange{zero_point},
        [result_type, results](mlir::OpBuilder& nested_b, mlir::Location loc,
                               mlir::Value iv, mlir::ValueRange loop_vars) {
          mlir::ImplicitLocOpBuilder b(loc, nested_b);
          auto acc = loop_vars[0];
          auto partial_result = b.create<mlir::memref::LoadOp>(results, iv);
          auto sum = b.create<mlir::zkir::elliptic_curve::AddOp>(
              result_type, partial_result, acc);
          b.create<mlir::scf::YieldOp>(mlir::ValueRange{sum});
        });

    return final_result.getResult(0);
  } else {
    return absl::InvalidArgumentError("bases is not a tensor");
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitDimensionsOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value input,
    absl::Span<const int64_t> source_dimensions) {
  pass_flag_.enable_linalg_to_parallel_loops = true;

  int64_t rank = instr->shape().rank();
  auto target_dimensions = [source_dimensions, rank]() {
    std::unordered_set<int64_t> source_set(source_dimensions.begin(),
                                           source_dimensions.end());
    std::vector<int64_t> target_dimensions;

    if (source_dimensions.empty()) {
      // If no source dims, return all dims in order
      for (int64_t i = 0; i < rank; ++i) {
        target_dimensions.push_back(i);
      }
      return target_dimensions;
    }

    int64_t pivot = source_dimensions[0];

    // Step 1: dims > pivot (after)
    for (int64_t i = pivot + 1; i < rank; ++i) {
      if (source_set.count(i) == 0) {
        target_dimensions.push_back(i);
      }
    }

    // Step 2: dims < pivot (before)
    for (int64_t i = 0; i < pivot; ++i) {
      if (source_set.count(i) == 0) {
        target_dimensions.push_back(i);
      }
    }

    return target_dimensions;
  };

  auto init = b.create<mlir::tensor::EmptyOp>(
      mlir_utils::ShapeToMlirTensorType(instr->shape(), b.getContext()),
      mlir::ValueRange{});

  switch (instr->opcode()) {
    case HloOpcode::kBroadcast: {
      if (ShapeUtil::IsScalar(instr->operand(0)->shape())) {
        const HloInstruction* input_instr = instr->operand(0);
        if (ShapeUtil::IsScalar(input_instr->shape())) {
          mlir::memref::LoadOp load =
              mlir::dyn_cast<mlir::memref::LoadOp>(input.getDefiningOp());
          if (!load) {
            return absl::InternalError("input is not a memref");
          }
          input = b.create<mlir::bufferization::ToTensorOp>(
              mlir_utils::ShapeToMlirTensorType(input_instr->shape(),
                                                b.getContext()),
              load.getMemref(),
              /*restrict=*/true,
              /*writable=*/false);
        }
      }
      auto broadcast =
          b.create<mlir::linalg::BroadcastOp>(input, init, target_dimensions());
      return broadcast.getResult()[0];
    }
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Unhandled dimensions op: %s", HloOpcodeString(instr->opcode())));
  }
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitMatrixVectorMultiplicationOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs,
    mlir::Value rhs) {
  if (LayoutUtil::IsDenseArray(instr->operand(0)->shape())) {
    return absl::UnimplementedError(
        "Dense matrix vector multiplication is not supported");
  }
  if (!LayoutUtil::IsCSRArray(instr->operand(0)->shape())) {
    return absl::UnimplementedError(
        "Only CSR matrix vector multiplication is supported");
  }
  pass_flag_.enable_linalg_to_parallel_loops = true;

  mlir::MLIRContext* ctx = lhs.getContext();
  auto result_type = mlir::cast<mlir::RankedTensorType>(
      mlir_utils::ShapeToMlirTensorType(instr->shape(), ctx));
  CHECK(result_type);
  llvm::SmallVector<int64_t> shapes;
  for (int64_t i = 0; i < instr->shape().dimensions_size(); ++i) {
    shapes.push_back(instr->shape().dimensions(i));
  }
  auto result =
      b.create<mlir::tensor::EmptyOp>(shapes, result_type.getElementType());

  auto d0 = b.getAffineDimExpr(0);
  auto d1 = b.getAffineDimExpr(1);

  auto generic_op = b.create<mlir::linalg::GenericOp>(
      /*resultTensorTypes=*/mlir::TypeRange{result_type},
      /*inputs=*/mlir::ValueRange{lhs, rhs},
      /*outputs=*/mlir::ValueRange{result},
      /*indexingMaps=*/
      llvm::SmallVector<mlir::AffineMap>{
          mlir::AffineMap::get(2, 0, {d0, d1}, ctx),
          mlir::AffineMap::get(2, 0, {d1}, ctx),
          mlir::AffineMap::get(2, 0, {d0}, ctx),
      },
      /*iteratorTypes=*/
      llvm::SmallVector<mlir::utils::IteratorType>{
          mlir::utils::IteratorType::parallel,
          mlir::utils::IteratorType::reduction,
      },
      /*doc=*/"matrix vector multiplication",
      /*libraryCall=*/mlir::StringRef(),
      [](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args) {
        mlir::ImplicitLocOpBuilder b(loc, builder);
        auto x = args[0];
        auto y = args[1];
        auto z = args[2];

        auto mul_op =
            b.create<mlir::sparse_tensor::BinaryOp>(x.getType(), x, y);
        {
          mlir::Region& overlap_region = mul_op.getOverlapRegion();
          mlir::Block* block = b.createBlock(&overlap_region);
          block->addArguments({x.getType(), y.getType()}, {loc, loc});
          b.setInsertionPointToStart(block);
          auto mul = b.create<mlir::zkir::field::MulOp>(block->getArgument(0),
                                                        block->getArgument(1));
          b.create<mlir::sparse_tensor::YieldOp>(mul);
        }

        b.setInsertionPointAfter(mul_op);
        auto reduce_op = b.create<mlir::sparse_tensor::ReduceOp>(
            mul_op.getType(), mul_op, z,
            /*identity=*/
            b.create<mlir::zkir::field::ConstantOp>(mul_op.getType(), 0));
        {
          mlir::Region& reduce_region = reduce_op.getRegion();
          mlir::Block* block = b.createBlock(&reduce_region);
          block->addArguments({mul_op.getType(), z.getType()}, {loc, loc});
          b.setInsertionPointToStart(block);
          auto add = b.create<mlir::zkir::field::AddOp>(block->getArgument(0),
                                                        block->getArgument(1));
          b.create<mlir::sparse_tensor::YieldOp>(add);
        }

        b.setInsertionPointAfter(reduce_op);
        b.create<mlir::linalg::YieldOp>(
            mlir::ValueRange{reduce_op.getOutput()});
      });
  return generic_op.getResult(0);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitDotOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value lhs,
    mlir::Value rhs) {
  int64_t rank0 = instr->operand(0)->shape().rank();
  int64_t rank1 = instr->operand(1)->shape().rank();

  if (rank0 == 1) {
    if (rank1 == 1) {
      return absl::UnimplementedError(
          "Dot op with vector and vector is not supported");
    } else if (rank1 == 2) {
      return absl::UnimplementedError(
          "Dot op with vector and matrix is not supported");
    }
  } else if (rank0 == 2) {
    if (rank1 == 1) {
      return EmitMatrixVectorMultiplicationOp(instr, b, lhs, rhs);
    } else if (rank1 == 2) {
      return absl::UnimplementedError(
          "Dot op with matrix and matrix is not supported");
    }
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Dot op with rank %d and rank %d is not supported", rank0, rank1));
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitSliceOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b, mlir::Value value,
    absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices,
    absl::Span<const int64_t> strides_in) {
  pass_flag_.enable_expand_strided_metadata = true;

  const Shape& shape = instr->shape();
  auto value_type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (value_type == nullptr) {
    return absl::InternalError("value is not a ranked tensor");
  }

  auto result_type = mlir::dyn_cast<mlir::RankedTensorType>(
      mlir_utils::ShapeToMlirTensorType(shape, b.getContext()));
  CHECK(result_type);

  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides;

  for (int64_t i = 0; i < shape.rank(); ++i) {
    offsets.push_back(b.getIndexAttr(start_indices[i]));
    sizes.push_back(b.getIndexAttr(limit_indices[i] - start_indices[i]));
    strides.push_back(b.getIndexAttr(strides_in[i]));
  }

  return b.create<mlir::tensor::ExtractSliceOp>(result_type, value, offsets,
                                                sizes, strides);
}

absl::StatusOr<mlir::Value> CpuKernelEmitter::EmitOp(
    const HloInstruction* instr, EmitterLocOpBuilder& b,
    absl::flat_hash_map<const HloInstruction*, mlir::Value>& values) {
  auto enable_flag = [this](PrimitiveType element_type) {
    if (primitive_util::IsFieldType(element_type)) {
      pass_flag_.enable_field_to_arith = true;
      return;
    } else if (primitive_util::IsEcPointType(element_type)) {
      pass_flag_.enable_elliptic_curve_to_field = true;
      pass_flag_.enable_elliptic_curve_to_llvm = true;
      switch (element_type) {
        case BN254_G2_AFFINE:
        case BN254_G2_JACOBIAN:
        case BN254_G2_XYZZ:
          pass_flag_.enable_ext_field_to_llvm = true;
          break;
        default:
          break;
      }
    }
  };

  enable_flag(instr->shape().element_type());
  if (instr->IsElementwise() && instr->shape().IsArray()) {
    pass_flag_.enable_elementwise_to_linalg = true;
  }

  switch (instr->opcode()) {
    case HloOpcode::kConvert:
    case HloOpcode::kInverse:
    case HloOpcode::kNegate: {
      return EmitUnaryOp(instr, b, values[instr->operand(0)]);
    }
    case HloOpcode::kAdd:
    case HloOpcode::kSubtract:
    case HloOpcode::kMultiply:
    case HloOpcode::kPower:
    case HloOpcode::kDivide: {
      enable_flag(instr->operand(0)->shape().element_type());
      enable_flag(instr->operand(1)->shape().element_type());
      return EmitBinaryOp(instr, b, values[instr->operand(0)],
                          values[instr->operand(1)]);
    }
    case HloOpcode::kFft: {
      if (instr->operand_count() == 1) {
        enable_flag(instr->operand(0)->shape().element_type());
        return EmitFftOp(instr, b, values[instr->operand(0)]);
      } else if (instr->operand_count() == 2) {
        enable_flag(instr->operand(0)->shape().element_type());
        enable_flag(instr->operand(1)->shape().element_type());
        return EmitFftOp(instr, b, values[instr->operand(0)],
                         values[instr->operand(1)]);
      } else {
        return absl::InternalError(
            "HloFftInstruction shouldn't have more than 2 arguments");
      }
    }
    case HloOpcode::kMsm: {
      enable_flag(instr->operand(0)->shape().element_type());
      enable_flag(instr->operand(1)->shape().element_type());
      return EmitMsmOp(instr, b, values[instr->operand(0)],
                       values[instr->operand(1)]);
    }
    case HloOpcode::kBroadcast: {
      enable_flag(instr->operand(0)->shape().element_type());
      return EmitDimensionsOp(instr, b, values[instr->operand(0)],
                              instr->dimensions());
    }
    case HloOpcode::kDot: {
      enable_flag(instr->operand(0)->shape().element_type());
      enable_flag(instr->operand(1)->shape().element_type());
      return EmitDotOp(instr, b, values[instr->operand(0)],
                       values[instr->operand(1)]);
    }
    case HloOpcode::kSlice: {
      enable_flag(instr->operand(0)->shape().element_type());
      return EmitSliceOp(instr, b, values[instr->operand(0)],
                         instr->slice_starts(), instr->slice_limits(),
                         instr->slice_strides());
    }

    default:
      return absl::UnimplementedError(
          absl::StrFormat("Unhandled opcode for IR emission: %s",
                          HloOpcodeString(instr->opcode())));
  }
}

}  // namespace zkx::cpu
