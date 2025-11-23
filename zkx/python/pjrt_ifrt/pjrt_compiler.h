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

#ifndef ZKX_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_
#define ZKX_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include "zkx/python/ifrt/compiler.h"
#include "zkx/python/ifrt/device_list.h"
#include "zkx/python/ifrt/executable.h"
#include "zkx/python/ifrt/program.h"
#include "zkx/python/ifrt/topology.h"

namespace zkx::ifrt {

class PjRtClient;

// Compiler that produces PjRt executables.
//
// TODO(hyeontaek): Move executable loading to `PjRtClient` and remove the
// requirement of `PjRtClient`, which will enable ahead-of-time compilation.
class PjRtCompiler final : public llvm::RTTIExtends<PjRtCompiler, Compiler> {
 public:
  explicit PjRtCompiler(PjRtClient* client) : client_(client) {}

  // Compiler implementation.

  ~PjRtCompiler() override = default;

  absl::StatusOr<LoadedExecutableRef> CompileAndLoad(
      std::unique_ptr<Program> program,
      std::unique_ptr<CompileOptions> options) override;

  absl::StatusOr<ExecutableRef> Compile(
      std::unique_ptr<Program> program, const Topology& topology,
      std::unique_ptr<CompileOptions> options) override;

  absl::Status IsExecutableVersionCompatible(
      const ExecutableVersion& executable_version,
      const DeviceListRef& devices) const override {
    return absl::UnimplementedError("Not implemented");
  }

  absl::StatusOr<LoadedExecutableRef> DeserializeLoadedExecutable(
      std::string_view serialized,
      std::unique_ptr<DeserializeExecutableOptions> options) override;

  static char ID;

 private:
  PjRtClient* client_;  // not owned
};

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_
