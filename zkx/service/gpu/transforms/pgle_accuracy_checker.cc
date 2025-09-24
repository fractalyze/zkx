/* Copyright 2024 The OpenXLA Authors.

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

#include "zkx/service/gpu/transforms/pgle_accuracy_checker.h"

#include "xla/tsl/platform/errors.h"

namespace zkx::gpu {

absl::StatusOr<bool> PGLEAccuracyChecker::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  TF_RETURN_IF_ERROR(pgle_estimator_.CheckAccuracy(*module));
  return false;
}

}  // namespace zkx::gpu
