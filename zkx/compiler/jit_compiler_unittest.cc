#include "zkx/compiler/jit_compiler.h"

#include <stdint.h>

#include <memory>

#include "gtest/gtest.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

namespace zkx::cpu {

using MainFuncType = int32_t (*)(int32_t, int32_t);

TEST(JitCompiler, CompileFromString) {
  const std::string_view ir = R"(
  define i32 @main(i32 %a, i32 %b) {
  entry:
    %sum = add i32 %a, %b
    ret i32 %sum
  }
  )";

  JitCompiler compiler;
  compiler.CompileFromString(ir);
  auto* main = compiler.GetMainFunction<MainFuncType>();
  // TODO(chokobole): Use random number.
  EXPECT_EQ(main(3, 4), 7);
}

TEST(JitCompiler, CompileFromFile) {
  JitCompiler compiler;
  compiler.CompileFromFile("zkx/compiler/tests/add.ll");
  auto* main = compiler.GetMainFunction<MainFuncType>();
  // TODO(chokobole): Use random number.
  EXPECT_EQ(main(3, 4), 7);
}

}  // namespace zkx::cpu
