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

#ifndef ZKX_ZKX_COMPILER_JIT_COMPILER_H_
#define ZKX_ZKX_COMPILER_JIT_COMPILER_H_

#include <memory>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace zkx::cpu {

// Jit compiler that compiles LLVM modules added to it into a FunctionLibrary.
// Jit-compiled function library will be backed by multiple dynamic libraries
// compiled from LLVM modules using LLVM ORC APIs.
class JitCompiler {
 public:
  JitCompiler() = default;
  JitCompiler(const JitCompiler& other) = delete;
  JitCompiler& operator=(const JitCompiler& other) = delete;
  ~JitCompiler() = default;

  void CompileFromString(std::string_view ir);
  // TODO(chokobole): If a specific file path type exists, consider renaming
  // this and the above to |Compile()|.
  void CompileFromFile(std::string_view ir_file);

  template <typename Func>
  auto GetMainFunction() {
    return symbol_.toPtr<Func>();
  }

 private:
  void Compile(std::unique_ptr<llvm::LLVMContext> context,
               std::unique_ptr<llvm::Module> module);

  std::unique_ptr<llvm::orc::LLJIT> jit_;
  llvm::orc::ExecutorAddr symbol_;
};

}  // namespace zkx::cpu

#endif  // ZKX_ZKX_COMPILER_JIT_COMPILER_H_
