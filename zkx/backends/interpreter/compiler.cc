/* Copyright 2017 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#include "zkx/backends/interpreter/compiler.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/interpreter/executable.h"
#include "zkx/backends/interpreter/platform_id.h"
#include "zkx/hlo/evaluator/hlo_evaluator.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/ir/hlo_module_group.h"
#include "zkx/hlo/pass/hlo_pass_pipeline.h"
#include "zkx/literal.h"
#include "zkx/service/computation_placer.h"
#include "zkx/service/custom_call_target_registry.h"
#include "zkx/service/dynamic_dimension_inference.h"
#include "zkx/service/executable.h"
#include "zkx/service/hlo_cost_analysis.h"
#include "zkx/service/layout_assignment.h"
#include "zkx/status_macros.h"
#include "zkx/stream_executor/platform.h"
#include "zkx/stream_executor/stream_executor.h"
#include "zkx/util.h"

namespace zkx::interpreter {

absl::Status InterpreterCompiler::RunHloOptimization(HloModule* hlo_module) {
  HloPassPipeline pipeline("Interpreter");

  // The TopkDecomposer generates a compare op with type=TOTALORDER and must
  // run before the ComparisonExpander which rewrites such comparisons.
  // TODO(chokobole): Uncomment this. Dependency: DynamicIndexSplitter
  // pipeline.AddPass<DynamicIndexSplitter>();
  pipeline.AddPass<LayoutAssignment>(
      hlo_module->mutable_entry_computation_layout());

  return pipeline.Run(hlo_module).status();
}

absl::StatusOr<std::unique_ptr<HloModule>> InterpreterCompiler::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* /*stream_exec*/,
    const CompileOptions& /*options*/) {
  VLOG(1) << "Run hlo passes on graph " << hlo_module->name();
  TF_RETURN_IF_ERROR(RunHloOptimization(hlo_module.get()));
  return std::move(hlo_module);
}

absl::StatusOr<std::unique_ptr<Executable>> InterpreterCompiler::RunBackend(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& /*options*/) {
  TF_RET_CHECK(stream_exec != nullptr);

  VLOG(1) << "Run backend " << hlo_module->name();

  TF_ASSIGN_OR_RETURN(
      DynamicDimensionInference dynamic_dimension_inference,
      DynamicDimensionInference::Run(
          hlo_module.get(),
          /*op_supports_dynamism_handler=*/[&](HloInstruction* hlo) {
            return OpDynamismSupport::kOptional;
          }));

  auto evaluator = std::make_unique<HloEvaluator>();
  // TODO(chokobole): Uncomment this. Dependency: custom_call_target
  // evaluator->set_custom_call_handler(HandleEvaluatorCustomCall);

  // Create executable from only the Hlo module.
  std::unique_ptr<Executable> executable =
      std::make_unique<InterpreterExecutable>(
          std::move(hlo_module), std::move(evaluator),
          std::move(dynamic_dimension_inference));

  return std::move(executable);
}

absl::StatusOr<std::vector<std::unique_ptr<Executable>>>
InterpreterCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_exec,
    const CompileOptions& options) {
  if (module_group->empty()) {
    return std::vector<std::unique_ptr<Executable>>();
  }
  if (module_group->size() > 1) {
    return absl::UnimplementedError(
        "Compilation of multiple HLO modules is not supported on Interpreter.");
  }
  if (stream_exec.size() != 1 || stream_exec[0].size() != 1) {
    return absl::UnimplementedError("Unexpected number of StreamExecutor's.");
  }
  auto hlo_modules = module_group->ConsumeModules();
  TF_ASSIGN_OR_RETURN(auto module, RunHloPasses(std::move(hlo_modules[0]),
                                                stream_exec[0][0], options));
  TF_ASSIGN_OR_RETURN(auto executable, RunBackend(std::move(module),
                                                  stream_exec[0][0], options));
  std::vector<std::unique_ptr<Executable>> ret;
  ret.push_back(std::move(executable));
  return std::move(ret);
}

absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
InterpreterCompiler::CompileAheadOfTime(
    std::unique_ptr<HloModuleGroup> module_group,
    const AotCompilationOptions& aot_options) {
  return absl::InvalidArgumentError(
      "AOT compilation not supported on Interpreter");
}

se::Platform::Id InterpreterCompiler::PlatformId() const {
  return se::interpreter::kZkxInterpreterPlatformId;
}

HloCostAnalysis::ShapeSizeFunction InterpreterCompiler::ShapeSizeBytesFunction()
    const {
  return InterpreterExecutable::ShapeSizeBytes;
}

}  // namespace zkx::interpreter

namespace {

bool InitModule() {
  zkx::Compiler::RegisterCompilerFactory(
      se::interpreter::kZkxInterpreterPlatformId, []() {
        return std::make_unique<zkx::interpreter::InterpreterCompiler>();
      });
  zkx::ComputationPlacer::RegisterComputationPlacer(
      se::interpreter::kZkxInterpreterPlatformId,
      []() { return std::make_unique<zkx::ComputationPlacer>(); });
  return true;
}

bool g_module_initialized = InitModule();

}  // namespace
