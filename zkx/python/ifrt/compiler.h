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

#ifndef ZKX_PYTHON_IFRT_COMPILER_H_
#define ZKX_PYTHON_IFRT_COMPILER_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include "zkx/python/ifrt/device_list.h"
#include "zkx/python/ifrt/executable.h"
#include "zkx/python/ifrt/executable_serdes.h"
#include "zkx/python/ifrt/program.h"
#include "zkx/python/ifrt/serdes.h"
#include "zkx/python/ifrt/topology.h"

namespace zkx::ifrt {

// Abstract options for compiling a program and load it as `LoadedExecutable`.
// Ideally, compile options should be present in the program being compiled to
// help static checking and completeness. This option structure is to express
// legacy compilation options that are not included in the program.
//
// TODO(hyeontaek): Make an new `LoadOptions` that is specific for loading.
struct CompileOptions : llvm::RTTIExtends<CompileOptions, Serializable> {
  static char ID;
};

// Represents a compiler that creates an `Executable` that can run a computation
// on devices.
//
// TODO(hyeontaek): All `Compiler` methods should take target information such
// as "Platform" or "Topology" that is not tied to a real hardware allocation,
// and return unloaded objects only. `Client` should take over the role of
// loading of compiled objects into the target low-level runtime and hardware to
// ready the them for execution. This will enable ahead-of-time compilation,
// better separation between compilation, loading, and serialization and
// deserialization.
class Compiler : public llvm::RTTIExtends<Compiler, llvm::RTTIRoot> {
 public:
  // Compiles `mlir_module` and returns a `LoadedExecutable`.
  // TODO(hyeontaek): Move executable loading to `Client`.
  absl::StatusOr<ExecutableRef> Compile(
      std::unique_ptr<Program> program,
      std::unique_ptr<CompileOptions> options) {
    return absl::UnimplementedError(
        "Compile returning ExecutableRef is not implemented.");
  }

  virtual absl::StatusOr<ExecutableRef> Compile(
      std::unique_ptr<Program> program, const Topology& topology,
      std::unique_ptr<CompileOptions> options) = 0;

  virtual absl::StatusOr<LoadedExecutableRef> CompileAndLoad(
      std::unique_ptr<Program> program,
      std::unique_ptr<CompileOptions> options) = 0;

  // Checks if an executable version is compatible if the executable would be
  // loaded onto specified `devices`.
  virtual absl::Status IsExecutableVersionCompatible(
      const ExecutableVersion& executable_version,
      const DeviceListRef& devices) const = 0;

  // Deserializes a serialized executable as produced by
  // `LoadedExecutable::Serialize()`. The compatibility of `serialized` is
  // implementation specific.
  // TODO(hyeontaek): Move executable loading to `Client`. Then, the user can
  // use standard IFRT deserialization instead of this custom deserialization
  // function.
  virtual absl::StatusOr<LoadedExecutableRef> DeserializeLoadedExecutable(
      std::string_view serialized,
      std::unique_ptr<DeserializeExecutableOptions> options) = 0;

  static char ID;
};

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_IFRT_COMPILER_H_
