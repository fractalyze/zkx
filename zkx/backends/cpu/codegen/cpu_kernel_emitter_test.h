#ifndef ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_

#include <string_view>

#include "absl/types/span.h"
#include "gtest/gtest.h"

#include "zkx/literal.h"

namespace zkx::cpu {

class CpuKernelEmitterTest : public testing::Test {
 public:
  void RunHlo(std::string_view hlo_string, absl::Span<Literal*> literals);
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_
