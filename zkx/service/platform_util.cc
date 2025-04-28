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
#include "absl/strings/str_join.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/base/logging.h"
#include "zkx/service/compiler.h"
#include "zkx/stream_executor/platform_manager.h"

namespace zkx {

// The name of the interpreter platform.
constexpr char kInterpreter[] = "interpreter";

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

absl::StatusOr<std::vector<se::Platform*>> GetSupportedPlatforms() {
  return se::PlatformManager::PlatformsWithFilter(
      [](const se::Platform* platform) {
        auto compiler_status = Compiler::GetForPlatform(platform);
        bool supported = compiler_status.ok();
        if (!supported) {
          LOG(INFO) << "platform " << platform->Name() << " present but no "
                    << "ZKX compiler available: "
                    << compiler_status.status().message();
        }
        return supported;
      });
}

}  // namespace

absl::StatusOr<std::string> PlatformUtil::CanonicalPlatformName(
    std::string_view platform_name) {
  return zkx::CanonicalPlatformName(platform_name);
}

absl::StatusOr<std::vector<se::Platform*>>
PlatformUtil::GetSupportedPlatforms() {
  // Gather all platforms which have an XLA compiler.
  return zkx::GetSupportedPlatforms();
}

absl::StatusOr<se::Platform*> PlatformUtil::GetDefaultPlatform() {
  TF_ASSIGN_OR_RETURN(auto platforms, GetSupportedPlatforms());

  se::Platform* platform = nullptr;
  if (platforms.empty()) {
    return absl::NotFoundError("no platforms found");
  } else if (platforms.size() == 1) {
    platform = platforms[0];
  } else if (platforms.size() == 2) {
    for (int i = 0; i < 2; i++) {
      if (absl::AsciiStrToLower(platforms[i]->Name()) == kInterpreter &&
          absl::AsciiStrToLower(platforms[1 - i]->Name()) != kInterpreter) {
        platform = platforms[1 - i];
        break;
      }
    }
  }
  if (platform != nullptr) {
    return platform;
  }

  // Multiple platforms present and we can't pick a reasonable default.
  std::string platforms_string = absl::StrJoin(
      platforms, ", ",
      [](std::string* out, const se::Platform* p) { out->append(p->Name()); });
  return absl::InvalidArgumentError(absl::StrFormat(
      "must specify platform because more than one platform (except for the "
      "interpreter platform) found: %s.",
      platforms_string));
}

// static
absl::StatusOr<se::Platform*> PlatformUtil::GetPlatform(
    std::string_view platform_name) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::PlatformManager::PlatformWithName(
                          zkx::CanonicalPlatformName(platform_name)));
  TF_RETURN_IF_ERROR(Compiler::GetForPlatform(platform).status());
  return platform;
}

}  // namespace zkx
