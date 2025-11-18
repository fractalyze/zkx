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

#include "zkx/service/cpu/cpu_options.h"

namespace {

const char* const kOptimizeForSizeCpuOption = "zkx_cpu_optimize_for_size";
const char* const kDisableSlpVectorizer = "zkx_cpu_disable_slp_vectorizer";
const char* const kDisableLoopUnrolling = "zkx_cpu_disable_loop_unrolling";

}  // namespace

namespace zkx::cpu::options {

bool OptimizeForSizeRequested(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().zkx_backend_extra_options();
  return extra_options_map.count(kOptimizeForSizeCpuOption) > 0;
}

bool SlpVectorizerDisabled(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().zkx_backend_extra_options();
  return extra_options_map.count(kDisableSlpVectorizer) > 0;
}

bool DisableLoopUnrolling(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().zkx_backend_extra_options();
  return extra_options_map.count(kDisableLoopUnrolling) > 0;
}

}  // namespace zkx::cpu::options
