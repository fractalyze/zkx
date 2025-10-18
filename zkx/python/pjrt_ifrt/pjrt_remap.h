/* Copyright 2024 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_PYTHON_PJRT_IFRT_PJRT_REMAP_H_
#define ZKX_PYTHON_PJRT_IFRT_PJRT_REMAP_H_

#include <vector>

#include "absl/status/statusor.h"

#include "xla/tsl/concurrency/ref_count.h"
#include "zkx/python/ifrt/array.h"
#include "zkx/python/ifrt/remap_plan.h"

namespace zkx::ifrt {

class PjRtCompatibleClient;

// Common implementation of `zkx::ifrt::Client::RemapArrays` for
// `PjRtCompatibleClient`.
absl::StatusOr<std::vector<ArrayRef>> PjRtCompatibleClientRemapArrays(
    PjRtCompatibleClient* client, const RemapPlan& plan,
    absl::Span<ArrayRef> arrays, ArrayCopySemantics semantics);

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_PJRT_IFRT_PJRT_REMAP_H_
