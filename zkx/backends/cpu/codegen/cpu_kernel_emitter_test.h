#ifndef ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"

#include "zkx/literal.h"
#include "zkx/service/hlo_runner.h"

namespace zkx::cpu {

class CpuKernelEmitterTest : public testing::Test {
 public:
  CpuKernelEmitterTest();

  absl::Status Compile(std::string_view hlo_string);
  absl::StatusOr<Literal> Run(absl::Span<Literal> literals);

 protected:
  HloRunner runner_;
  std::unique_ptr<OpaqueExecutable> opaque_executable_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_
