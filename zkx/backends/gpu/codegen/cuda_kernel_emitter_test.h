#ifndef ZKX_BACKENDS_GPU_CODEGEN_CUDA_KERNEL_EMITTER_TEST_H_
#define ZKX_BACKENDS_GPU_CODEGEN_CUDA_KERNEL_EMITTER_TEST_H_

#include "zkx/backends/kernel_emitter_test.h"

namespace zkx::gpu {

class CudaKernelEmitterTest : public KernelEmitterTest {
 public:
  CudaKernelEmitterTest() : KernelEmitterTest("cuda") {}
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_CUDA_KERNEL_EMITTER_TEST_H_
