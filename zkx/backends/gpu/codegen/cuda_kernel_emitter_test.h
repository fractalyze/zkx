/* Copyright 2025 The ZKX Authors.

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
