#ifndef ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_

#include "zkx/backends/kernel_emitter_test.h"

namespace zkx::cpu {

class CpuKernelEmitterTest : public KernelEmitterTest {
 public:
  CpuKernelEmitterTest() : KernelEmitterTest("cpu") {}
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_
