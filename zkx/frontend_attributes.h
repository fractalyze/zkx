/* Copyright 2017 The OpenXLA Authors.
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

#ifndef ZKX_FRONTEND_ATTRIBUTES_H_
#define ZKX_FRONTEND_ATTRIBUTES_H_

#include "zkx/hlo/ir/hlo_instruction.h"

namespace zkx {

// Set frontend attribute on `instruction` which indices that in-place
// `instruction` has disjoint read/write buffer regions.
void SetDisjointReadWriteRegionsAttr(HloInstruction* instruction);

// Returns `true` if in-place `instruction` has disjoint read/write buffer
// regions.
bool HasDisjointReadWriteRegionsAttr(HloInstruction* instruction);

}  // namespace zkx

#endif  // ZKX_FRONTEND_ATTRIBUTES_H_
