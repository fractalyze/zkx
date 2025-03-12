/* Copyright 2017 The OpenXLA Authors.

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

#include "zkx/service/platform_util.h"

#include "absl/strings/ascii.h"

namespace zkx {
namespace {

std::string CanonicalPlatformName(std::string_view platform_name) {
  std::string lowercase_platform_name = absl::AsciiStrToLower(platform_name);
  // "cpu" and "host" mean the same thing.
  if (lowercase_platform_name == "cpu") {
    return "host";
  }
  // When configured on CUDA, "gpu" and "cuda" mean the same thing.
  // When configured on ROCm, "gpu" and "rocm" mean the same thing.
  // When configured on SYCL, "gpu" and "sycl" mean the same thing.
  if (lowercase_platform_name == "gpu") {
#if ZKX_USE_ROCM
    return "rocm";
#elif ZKX_USE_SYCL
    return "sycl";
#else
    return "cuda";
#endif
  }
  return lowercase_platform_name;
}

}  // namespace

absl::StatusOr<std::string> PlatformUtil::CanonicalPlatformName(
    std::string_view platform_name) {
  return zkx::CanonicalPlatformName(platform_name);
}

}  // namespace zkx
