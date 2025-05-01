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

#ifndef ZKX_SERVICE_CPU_CPU_COMPILER_H_
#define ZKX_SERVICE_CPU_CPU_COMPILER_H_

#include <memory>

#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_schedule.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/cpu/cpu_executable.h"

namespace zkx::cpu {

// CPU-targeting implementation of the ZKX Compiler interface.
//
// The compiler translates ZKX HLO code into LLVM IR and uses LLVM's JIT
// infrastructure to create an executable "blob" that can then be returned
// wrapped in CpuExecutable and actually invoked.
class CpuCompiler {
 public:
  CpuCompiler() = default;
  ~CpuCompiler() = default;

  absl::StatusOr<std::unique_ptr<CpuExecutable>> RunBackend(
      std::unique_ptr<HloModule> module);

 private:
  absl::StatusOr<HloSchedule> CreateHloSchedule(
      const HloModule& hlo_module) const;

  absl::StatusOr<std::unique_ptr<BufferAssignment>> CreateBufferAssignment(
      const HloModule& module) const;

  absl::StatusOr<std::unique_ptr<CpuExecutable>> CompileCpuExecutable(
      std::unique_ptr<HloModule> module);

  CpuCompiler(const CpuCompiler&) = delete;
  CpuCompiler& operator=(const CpuCompiler&) = delete;
};

}  // namespace zkx::cpu

#endif  // ZKX_SERVICE_CPU_CPU_COMPILER_H_
