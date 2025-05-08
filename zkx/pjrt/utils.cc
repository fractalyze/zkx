/* Copyright 2020 The OpenXLA Authors.

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

#include "zkx/pjrt/utils.h"

#include <algorithm>
#include <cstdlib>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/numbers.h"

#include "xla/tsl/platform/cpu_info.h"
#include "zkx/primitive_util.h"

namespace zkx {

int DefaultThreadPoolSize() {
  // Google's CI system exposes an environment variable NPROC that describes
  // a CPU reservation for tests.
  // TODO(phawkins): expose a better thought-out set of knobs to control
  // parallelism.
  for (const char* nproc_env : {"PJRT_NPROC", "NPROC"}) {
    const char* nproc_str = std::getenv(nproc_env);
    int nproc = 0;
    if (nproc_str && absl::SimpleAtoi(nproc_str, &nproc)) {
      return std::max(0, nproc);
    }
  }
  return tsl::port::MaxParallelism();
}

bool HasMajorToMinorLayout(PrimitiveType type, absl::Span<const int64_t> dims,
                           absl::Span<const int64_t> byte_strides) {
  CHECK_EQ(dims.size(), byte_strides.size());
  // If the array is size 0, the strides are irrelevant.
  if (absl::c_find(dims, 0) != dims.end()) {
    return true;
  }
  int64_t stride = primitive_util::ByteWidth(type);
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    // If a dimension is of size 1, its stride is irrelevant.
    if (dims[i] != 1) {
      if (byte_strides[i] != stride) {
        return false;
      }
      stride *= dims[i];
    }
  }
  return true;
}

}  // namespace zkx
