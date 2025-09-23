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
#include "zkx/frontend_attributes.h"

namespace zkx {
namespace {

// Attribute which indicates that an in-place instruction has disjoint read
// and write regions w.r.t aliased input/output buffers.
constexpr char kZkxDisjointReadWriteRegions[] =
    "_zkx_disjoint_read_write_regions";

}  // namespace

void SetDisjointReadWriteRegionsAttr(HloInstruction* instruction) {
  instruction->set_frontend_attribute(kZkxDisjointReadWriteRegions, "true");
}

bool HasDisjointReadWriteRegionsAttr(HloInstruction* instruction) {
  return instruction->frontend_attributes().map().contains(
      kZkxDisjointReadWriteRegions);
}

}  // namespace zkx
