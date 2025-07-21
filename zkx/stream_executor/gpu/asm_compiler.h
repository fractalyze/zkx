/* Copyright 2019 The OpenXLA Authors.

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

#ifndef ZKX_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_
#define ZKX_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/status/statusor.h"

#include "zkx/stream_executor/namespace_alias.h" // IWYU pragma: keep

namespace stream_executor {

struct HsacoImage {
  std::string gfx_arch;
  std::vector<uint8_t> bytes;
};

// Bundles the GPU machine code (HSA Code Object) and returns the resulting
// binary (i.e. a fatbin) as a byte array.
absl::StatusOr<std::vector<uint8_t>> BundleGpuAsm(
    const std::vector<HsacoImage>& images, const std::string& rocm_root_dir);

}  // namespace stream_executor

#endif  // ZKX_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_
