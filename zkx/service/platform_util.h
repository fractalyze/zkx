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

#ifndef ZKX_SERVICE_PLATFORM_UTIL_H_
#define ZKX_SERVICE_PLATFORM_UTIL_H_

#include <string>

#include "absl/status/statusor.h"

namespace zkx {

// Utilities for querying platforms and devices used by ZKX.
class PlatformUtil {
 public:
  // Returns the canonical name of the underlying platform.
  //
  // This is needed to differentiate if for given platform like GPU or CPU
  // there are multiple implementations. For example, GPU platform may be
  // cuda(Nvidia) or rocm(AMD)
  static absl::StatusOr<std::string> CanonicalPlatformName(
      std::string_view platform_name);

 private:
  PlatformUtil(const PlatformUtil&) = delete;
  PlatformUtil& operator=(const PlatformUtil&) = delete;
};

}  // namespace zkx

#endif  // ZKX_SERVICE_PLATFORM_UTIL_H_
