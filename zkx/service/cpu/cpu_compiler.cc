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

#include "zkx/service/cpu/cpu_compiler.h"

#include <functional>
#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "xla/tsl/platform/cpu_info.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/thread_pool.h"
#include "zkx/backends/cpu/codegen/cpu_features.h"
#include "zkx/backends/cpu/codegen/jit_compiler.h"
#include "zkx/base/logging.h"
#include "zkx/cpu_function_runtime.h"
#include "zkx/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "zkx/service/buffer_value.h"
#include "zkx/service/cpu/cpu_options.h"
#include "zkx/service/cpu/thunk_emitter.h"
#include "zkx/shape_util.h"
#include "zkx/util.h"

namespace zkx::cpu {

namespace {

// A module identifier (prefix) for emitted LLVM modules.
constexpr std::string_view kZkxModuleIdentifier = "__compute_module";

// Align buffers to XLA:CPU minimal alignment.
int64_t memory_alignment(LogicalBuffer::Color) {
  return cpu_function_runtime::MinAlign();
}

int64_t ShapeSizeBytes(const Shape& shape) {
  // On the cpu, opaques are pointers.
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  if (shape.is_static() || shape.IsTuple()) {
    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }
  // Each dynamic dimension size is represented as a S32.
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions_size();
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*)) + metadata_size;
}

// TODO(chokobole): Remove this. Dependency: Compiler
std::function<int64_t(const BufferValue&)> BufferSizeBytesFunction() {
  return
      [](const BufferValue& buffer) { return ShapeSizeBytes(buffer.shape()); };
}

// Returns a global (per-process) thread pool for XLA CPU compilation tasks.
tsl::thread::ThreadPool* GetCompilationThreadPool() {
  // LLVM compilation has a lot of memory-bound pointer chasing and not
  // so much CPU-bound work. Based on profiling a few examples, 32 threads seems
  // to be enough to achieve maximum parallel compilation speedup.
  static constexpr int kMaxCompilationThreads = 32;
  static auto* thread_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), "zkx-cpu-llvm-codegen",
      std::min(kMaxCompilationThreads, tsl::port::MaxParallelism()));
  return thread_pool;
}

// Returns task runner that uses the global compilation thread pool.
JitCompiler::TaskRunner GetCompilationTaskRunner() {
  return [](JitCompiler::Task task) {
    GetCompilationThreadPool()->Schedule(std::move(task));
  };
}

inline void VlogMaxIsa(std::string_view max_cpu_isa) {
  // TODO(chokobole): Uncomment this. Dependency: VLOG_IS_ON
  // if (VLOG_IS_ON(1) && !max_cpu_isa.empty()) {
  if (!max_cpu_isa.empty()) {
    if (tsl::port::IsX86CPU()) {
      VLOG(1) << "`zkx_cpu_max_isa` is set. Will not use features newer than: "
              << max_cpu_isa;
    } else {
      VLOG(1) << "`zkx_cpu_max_isa` is set to `" << max_cpu_isa
              << "`. This flag is not supported on non-x86 CPUs yet.";
    }
  }
}

absl::StatusOr<CpuExecutable::ConstantAllocation> LiteralToConstantAllocation(
    BufferAllocation::Index index, const Literal& literal) {
  // TODO(ezhulenev): This code is almost identical to code in XLA:GPU, we
  // should standardize it. See `xla/service/gpu/ir_emission_utils.cc`.
  PrimitiveType element_type = literal.shape().element_type();
  if (!primitive_util::IsArrayType(element_type)) {
    return absl::InternalError(
        "Only array literals can be converted to constant allocations");
  }

  int64_t size_bytes = literal.size_bytes();
  const void* untyped_data = literal.untyped_data();

  // Pack sub-byte types into an XLA storage format.
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    int bit_width = primitive_util::BitWidth(element_type);
    int packed_size_bytes = CeilOfRatio<int64_t>(size_bytes, 8 / bit_width);

    // Use Literal as a storage for packed data as it allocates underlying
    // buffer with correct alignment. Keep it allocated on heap to avoid
    // capturing stack address that will be invalidated by a move below.
    auto packed = std::make_unique<Literal>(
        ShapeUtil::MakeShape(U8, {packed_size_bytes}));

    PackIntN(
        bit_width,
        absl::MakeSpan(reinterpret_cast<const char*>(untyped_data), size_bytes),
        absl::MakeSpan(reinterpret_cast<char*>(packed->untyped_data()),
                       packed->size_bytes()));

    return CpuExecutable::ConstantAllocation{index, std::move(packed)};
  }

  // Create a constant allocation from the literal's untyped data.
  return CpuExecutable::ConstantAllocation{
      index, absl::Span<const uint8_t>(
                 reinterpret_cast<const uint8_t*>(untyped_data), size_bytes)};
}

// Creates a vector of constant allocations from the given buffer assignment.
absl::StatusOr<std::vector<CpuExecutable::ConstantAllocation>>
CreateConstantAllocations(const BufferAssignment& assignment) {
  std::vector<CpuExecutable::ConstantAllocation> constants;

  for (const BufferAllocation& allocation : assignment.Allocations()) {
    if (!allocation.is_constant()) {
      continue;
    }

    // Find the constant instruction defining the value for allocation.
    HloInstruction* const_instr = nullptr;
    for (const auto& [value, _] : allocation.assigned_buffers()) {
      // Multiple aliasing instructions can share the allocation, we need to
      // find the original constant instruction that defines the value.
      if (value->instruction()->opcode() == HloOpcode::kConstant) {
        if (const_instr != nullptr) {
          return absl::InternalError(
              absl::StrCat("Multiple constant instructions define buffer ",
                           allocation.ToString()));
        }
        const_instr = value->instruction();
      }
    }
    if (const_instr == nullptr) {
      return absl::InternalError(
          absl::StrCat("Could not find constant instruction defining buffer ",
                       allocation.ToString()));
    }

    VLOG(3) << "Create constant allocation for index " << allocation.index()
            << " from constant literal " << const_instr->name()
            << "; shape=" << const_instr->literal().shape();
    TF_ASSIGN_OR_RETURN(constants.emplace_back(),
                        LiteralToConstantAllocation(allocation.index(),
                                                    const_instr->literal()));
  }

  return constants;
}

}  // namespace

absl::StatusOr<std::unique_ptr<CpuExecutable>>
CpuCompiler::CompileCpuExecutable(std::unique_ptr<HloModule> module) {
  // TODO(chokobole): Uncomment this. Dependency: Profiler
  // TraceMe trace([&] {
  //   return TraceMeEncode("CpuCompiler::CompileCpuExecutable",
  //                        {{"name", module->name()}});
  // });

  // TODO(chokobole): Uncomment this. Dependency: GetIRModuleHooks
  // ModuleHook pre_optimization_ir_hook;
  // ModuleHook post_optimization_ir_hook;
  // std::tie(pre_optimization_ir_hook, post_optimization_ir_hook) =
  //     GetIRModuleHooks(*module, user_pre_optimization_hook_,
  //                      user_post_optimization_hook_);

  // Compile must be thread-safe so create a new LLVM context for the module.
  mlir::DialectRegistry registry;
  auto mlir_context = std::make_unique<mlir::MLIRContext>(registry);
  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module =
      std::make_unique<llvm::Module>(kZkxModuleIdentifier, *llvm_context);

  const DebugOptions& debug_options = module->config().debug_options();

  // We collect compiled object files (machine code) so we can export
  // CpuExecutable to an AOT compilation result.
  std::vector<std::string> obj_files;

  // We split LLVM module and distribute it across separate DyLibs to enable
  // parallel compilation at run time.
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: DebugOptions::zkx_cpu_parallel_codegen_split_count
  // clang-format on
  // size_t parallel_codegen_split_count =
  //     debug_options.zkx_cpu_parallel_codegen_split_count();
  VlogMaxIsa(debug_options.zkx_cpu_max_isa());

  const HloModuleConfig& config = module->config();

  // Options for compiling LLVM IR to machine code.
  IrCompiler::Options ir_compiler_options{
      /*optimization_level=*/IrCompiler::GetCodeGenOptLevel(config),
      /*optimize_for_size=*/options::OptimizeForSizeRequested(config),
      /*disable_expensive_passes=*/
      debug_options.zkx_llvm_disable_expensive_passes(),
      /*slp_vectorizer_disabled=*/options::SlpVectorizerDisabled(config),
      /*disable_loop_unrolling=*/options::DisableLoopUnrolling(config),
  };

  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: CreateOrcJITPostCompilationHook
  // clang-format on
  // Compiler hooks to intercept compiled LLVM IR modules.
  // IrCompiler::CompilationHooks ir_compiler_hooks{
  //     pre_optimization_ir_hook,
  //     post_optimization_ir_hook,
  //     CreateOrcJITPostCompilationHook(module.get(), &obj_files),
  // };

  // Definition generator to link with ZKX:CPU host runtime symbols.
  // TODO(chokobole): Uncomment this. Dependency: RuntimeSymbolGenerator
  // ExecutionEngine::DefinitionGenerator definition_generator =
  //     [](const llvm::DataLayout& data_layout) {
  //       return std::make_unique<RuntimeSymbolGenerator>(data_layout);
  //     };

  // Options for orchestrating the JIT compilation process.
  // JitCompiler::Options jit_compiler_options{
  //     std::move(ir_compiler_options),
  //     std::move(ir_compiler_hooks),
  //     /*num_dylibs=*/parallel_codegen_split_count,
  //     /*definition_generator=*/std::move(definition_generator),
  //     /*max_cpu_isa=*/CpuFeatureFromString(debug_options.xla_cpu_max_isa()),
  // };
  JitCompiler::Options jit_compiler_options;
  jit_compiler_options.ir_compiler_options = std::move(ir_compiler_options);
  jit_compiler_options.max_cpu_feature =
      CpuFeatureFromString(debug_options.zkx_cpu_max_isa());

  TF_ASSIGN_OR_RETURN(JitCompiler jit_compiler,
                      JitCompiler::Create(llvm::TargetOptions(),
                                          std::move(jit_compiler_options),
                                          GetCompilationTaskRunner()));

  // TODO(chokobole): Uncomment this. Dependency: CreateHloProfilingArtifacts
  // absl::flat_hash_map<const HloInstruction*, int64_t>
  //     instruction_to_profile_idx;
  // absl::flat_hash_map<const HloComputation*, int64_t>
  //     computation_to_profile_idx;
  // std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map;
  // std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data;
  // if (module->config().hlo_profiling_enabled()) {
  //   TF_RETURN_IF_ERROR(CreateHloProfilingArtifacts(
  //       *module, &instruction_to_profile_idx, &computation_to_profile_idx,
  //       &hlo_profile_index_map, &hlo_profile_printer_data));
  // }

  // Cache these flags here since we'll want to access them after the module's
  // ownership is std::moved.
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: DebugOptions::zkx_embed_ir_in_executable
  // clang-format on
  // const bool embed_ir_in_executable =
  //     debug_options.zkx_embed_ir_in_executable();

  TF_ASSIGN_OR_RETURN(HloSchedule schedule, CreateHloSchedule(*module));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> assignment,
                      CreateBufferAssignment(*module));
  // TODO(chokobole): Uncomment this. Dependency: DumpHloModuleIfEnabled
  // DumpHloModuleIfEnabled(*module, *assignment,
  //                        absl::StrCat("cpu_", kAfterOptimizationsDumpName));

  // Thunk emitter is responsible for building a Thunk sequence that will
  // resolved kernels in the compiled LLVM module and execute them together
  // with Thunks implemented as library calls (e.g. oneDNN or Eigen).
  ThunkEmitter thunk_emitter(assignment.get(), mlir_context.get());
  TF_ASSIGN_OR_RETURN(ThunkSequence thunks,
                      thunk_emitter.EmitEntryComputation(*module));

  // Collect compiled symbols from all LLVM module parts.
  std::vector<FunctionLibrary::Symbol> compiled_symbols;
  for (auto& [name, module] : thunk_emitter.kernels()) {
    compiled_symbols.push_back(
        FunctionLibrary::Sym<FunctionLibrary::Kernel>(name));
    // clang-format off
    // TODO(chokobole): Uncomment this. Dependency: symbol_type_id_to_function_type_id
    // clang-format on
    // symbol_type_id_to_function_type_id.emplace(compiled_symbols.back().type_id,
    //                                            SymbolProto::KERNEL);
    // TODO(chokobole): Uncomment this. Dependency: kernel_dylib_index
    // TF_CHECK_OK(jit_compiler.AddModule(std::move(module),
    // kernel_dylib_index));
    TF_CHECK_OK(jit_compiler.AddModule(std::move(module)));
    // Simply roundrobin the kernel dylibs
    // kernel_dylib_index = (kernel_dylib_index + 1) % num_parts;
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<FunctionLibrary> function_library,
                      std::move(jit_compiler).Compile(compiled_symbols));

  // Create constant allocations from the buffer assignment.
  TF_ASSIGN_OR_RETURN(std::vector<CpuExecutable::ConstantAllocation> constants,
                      CreateConstantAllocations(*assignment));

  TF_ASSIGN_OR_RETURN(
      auto cpu_executable,
      CpuExecutable::Create(std::move(function_library), std::move(assignment),
                            std::move(module), std::move(thunks),
                            std::move(constants)));

  return std::move(cpu_executable);
  // TODO(chokobole): Implement other branch.
  // if (!module->config().debug_options().xla_cpu_use_thunk_runtime()) {
}

absl::StatusOr<std::unique_ptr<CpuExecutable>> CpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module) {
  // TODO(chokobole): Uncomment this. Dependency: Profiler
  // TraceMe trace([&] {
  //   return TraceMeEncode("CpuCompiler::RunBackend", {{"name",
  //   module->name()}});
  // });

  VLOG(1) << "Compiling: " << module->name();
  // TODO(chokobole): Uncomment this. Dependency: RecordCpuCompilerStacktrace
  // RecordCpuCompilerStacktrace();
  // TODO(chokobole): Uncomment this. Dependency: XLA_SCOPED_LOGGING_TIMER
  // XLA_SCOPED_LOGGING_TIMER(
  //     absl::StrFormat("Compiling [%s] for CPU using JIT", module->name()));

  // TODO(chokobole): Uncomment this. Dependency: SlowCompilationAlarm
  // std::string slow_compilation_msg =
  //     absl::StrCat("Compiling module ", module->name());
  // auto slow_compile_alarm = SlowCompilationAlarm(slow_compilation_msg);
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: llvm_ir::LLVMCommandLineOptionsLock
  // clang-format on
  // auto llvm_options = llvm_ir::ExtractXlaBackendExtraOptions(
  //     module->config().debug_options().xla_backend_extra_options());
  // llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_options);

  std::unique_ptr<CpuExecutable> cpu_executable;
  TF_ASSIGN_OR_RETURN(cpu_executable, CompileCpuExecutable(std::move(module)));

  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: CpuExecutable::set_debug_info
  // clang-format on
  // cpu_executable->set_debug_info(
  //     cpu_executable->buffer_assignment().StatsString(
  //         /*report_total_fragmentation=*/true));
  VLOG(1) << "Compilation finished";
  return std::unique_ptr<CpuExecutable>(std::move(cpu_executable));
}

absl::StatusOr<HloSchedule> CpuCompiler::CreateHloSchedule(
    const HloModule& hlo_module) const {
  // Select a memory scheduler optimized for concurrency vs minimal memory.
  auto scheduler = hlo_module.config()
                           .debug_options()
                           .zkx_cpu_enable_concurrency_optimized_scheduler()
                       ? BFSMemoryScheduler
                       : DFSMemoryScheduler;

  // Select an order for emitting the HLO instructions for each
  // computation. Using this sequence enables tighter buffer liveness analysis
  // and reduced memory usage (as compared to using `DependencyHloOrdering`).
  return ScheduleModule(&hlo_module, BufferSizeBytesFunction(),
                        ComputationSchedulerToModuleScheduler(scheduler));
}

absl::StatusOr<std::unique_ptr<BufferAssignment>>
CpuCompiler::CreateBufferAssignment(const HloModule& module) const {
  // Run buffer allocation on the HLO graph.
  return BufferAssigner::Run(
      &module, std::make_unique<SequentialHloOrdering>(module.schedule()),
      BufferSizeBytesFunction(), memory_alignment,
      /*allocate_buffers_for_constants=*/true);
}

}  // namespace zkx::cpu
