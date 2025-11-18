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

#ifndef ZKX_BACKENDS_CPU_RUNTIME_ALL_TO_ALL_THUNK_H_
#define ZKX_BACKENDS_CPU_RUNTIME_ALL_TO_ALL_THUNK_H_

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"

#include "zkx/backends/cpu/runtime/collective_thunk.h"

namespace zkx::cpu {

class AllToAllThunk final : public CollectiveThunk {
 public:
  static absl::StatusOr<std::unique_ptr<AllToAllThunk>> Create(
      Info info, OpParams op_params, OpBuffers op_buffers,
      OpResources op_resources) {
    return absl::WrapUnique(
        new AllToAllThunk(std::move(info), std::move(op_params),
                          std::move(op_buffers), std::move(op_resources)));
  }

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

 private:
  AllToAllThunk(Info info, OpParams op_params, OpBuffers op_buffers,
                OpResources op_resources)
      : CollectiveThunk(Kind::kAllToAll, std::move(info), std::move(op_params),
                        std::move(op_buffers), std::move(op_resources)) {}
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_RUNTIME_ALL_TO_ALL_THUNK_H_
