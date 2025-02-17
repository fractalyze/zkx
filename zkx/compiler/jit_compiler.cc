#include "zkx/compiler/jit_compiler.h"

#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

namespace zkx::cpu {

absl::once_flag initialize_llvm_flag;

namespace {

// Initialize LLVM the first time `JitCompiler` is created.
void InitializeLLVMTarget() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
}

}  // namespace

void JitCompiler::CompileFromString(std::string_view ir) {
  std::unique_ptr<llvm::LLVMContext> context(new llvm::LLVMContext);

  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::MemoryBuffer> buffer =
      llvm::MemoryBuffer::getMemBuffer(ir);
  std::unique_ptr<llvm::Module> module =
      llvm::parseIR(buffer->getMemBufferRef(), err, *context);
  // TODO(chokobole): emit error message using |err|.
  CHECK(module);

  Compile(std::move(context), std::move(module));
}

void JitCompiler::CompileFromFile(std::string_view ir_file) {
  std::unique_ptr<llvm::LLVMContext> context(new llvm::LLVMContext);

  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(ir_file, err, *context);
  // TODO(chokobole): emit error message using |err|.
  CHECK(module);

  Compile(std::move(context), std::move(module));
}

void JitCompiler::Compile(std::unique_ptr<llvm::LLVMContext> context,
                          std::unique_ptr<llvm::Module> module) {
  absl::call_once(initialize_llvm_flag, InitializeLLVMTarget);

  llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> jit =
      llvm::orc::LLJITBuilder().create();
  CHECK(jit) << "failed creating JIT: " << llvm::toString(jit.takeError());

  llvm::Error err = std::move(jit).moveInto(jit_);
  CHECK(!err) << "failed moving jit to jit_: "
              << llvm::toString(std::move(err));

  llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(context));
  err = jit_->addIRModule(std::move(tsm));
  CHECK(!err) << "failed adding ir module: " << llvm::toString(std::move(err));

  llvm::Expected<llvm::orc::ExecutorAddr> symbol = jit_->lookup("main");
  CHECK(symbol) << "failed finding main function";

  err = std::move(symbol).moveInto(symbol_);
  CHECK(!err) << "failed moving symbol to symbol_: "
              << llvm::toString(std::move(err));
}

}  // namespace zkx::cpu
