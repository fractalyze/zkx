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

#ifndef ZKX_BACKENDS_CPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_
#define ZKX_BACKENDS_CPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"

#include "zkx/backends/cpu/runtime/collective_thunk.h"

namespace zkx::cpu {

class CollectivePermuteThunk final : public CollectiveThunk {
 public:
  using SourceTargetPair = std::pair<int64_t, int64_t>;

  static absl::StatusOr<std::unique_ptr<CollectivePermuteThunk>> Create(
      Info info, OpParams op_params, OpBuffers op_buffers,
      OpResources op_resources,
      absl::Span<const SourceTargetPair> source_target_pairs) {
    return absl::WrapUnique(new CollectivePermuteThunk(
        std::move(info), std::move(op_params), std::move(op_buffers),
        std::move(op_resources), source_target_pairs));
  }

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  const std::vector<SourceTargetPair>& source_target_pairs() const {
    return source_target_pairs_;
  }

 private:
  CollectivePermuteThunk(Info info, OpParams op_params, OpBuffers op_buffers,
                         OpResources op_resources,
                         absl::Span<const SourceTargetPair> source_target_pairs)
      : CollectiveThunk(Kind::kCollectivePermute, std::move(info),
                        std::move(op_params), std::move(op_buffers),
                        std::move(op_resources)),
        source_target_pairs_(source_target_pairs.begin(),
                             source_target_pairs.end()) {}

  std::vector<SourceTargetPair> source_target_pairs_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_
