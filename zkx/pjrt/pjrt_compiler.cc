/* Copyright 2022 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/pjrt/pjrt_compiler.h"

#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/debugging/leak_check.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"

namespace zkx {
namespace {

ABSL_CONST_INIT absl::Mutex g_registry_mutex(absl::kConstInit);
absl::flat_hash_map<std::string, std::unique_ptr<PjRtCompiler>>*
CompilerRegistry() {
  static auto* compiler_registry = absl::IgnoreLeak(
      new absl::flat_hash_map<std::string, std::unique_ptr<PjRtCompiler>>());
  return compiler_registry;
}

}  // namespace

void PjRtRegisterCompiler(std::string_view platform_name,
                          std::unique_ptr<PjRtCompiler> compiler) {
  CHECK(compiler != nullptr);
  absl::MutexLock l(&g_registry_mutex);
  auto* compiler_registry = CompilerRegistry();
  CHECK(!compiler_registry->contains(platform_name));
  (*compiler_registry)[platform_name] = std::move(compiler);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, const ZkxComputation& computation,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  auto topology_compiler = topology.compiler();
  // TODO(chokobole): Uncomment this. Dependency: ScopedMetricHelper
  // ScopedMetricHelper
  // helper(metrics::kPjrtCompilerCompileComputationMetricName);
  if (topology_compiler.has_value()) {
    return (*topology_compiler)
        ->Compile(std::move(options), computation, topology, client);
  }
  absl::ReaderMutexLock l(&g_registry_mutex);
  const auto* compiler_registry = CompilerRegistry();
  auto it = compiler_registry->find(topology.platform_name());
  if (it == compiler_registry->end()) {
    return absl::NotFoundError(absl::StrCat(
        "No compiler registered for platform ", topology.platform_name()));
  }
  return it->second->Compile(std::move(options), computation, topology, client);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCompile(
    CompileOptions options, mlir::ModuleOp module,
    const PjRtTopologyDescription& topology, PjRtClient* client) {
  auto topology_compiler = topology.compiler();
  // TODO(chokobole): Uncomment this. Dependency: ScopedMetricHelper
  // ScopedMetricHelper helper(metrics::kPjrtCompilerCompileModuleMetricName);
  if (topology_compiler.has_value()) {
    return (*topology_compiler)
        ->Compile(std::move(options), module, topology, client);
  }
  absl::ReaderMutexLock l(&g_registry_mutex);
  const auto* compiler_registry = CompilerRegistry();
  auto it = compiler_registry->find(topology.platform_name());
  if (it == compiler_registry->end()) {
    return absl::NotFoundError(absl::StrCat(
        "No compiler registered for platform ", topology.platform_name()));
  }
  return it->second->Compile(std::move(options), module, topology, client);
}

}  // namespace zkx
