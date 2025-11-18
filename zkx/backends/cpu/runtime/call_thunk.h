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

#ifndef ZKX_BACKENDS_CPU_RUNTIME_CALL_THUNK_H_
#define ZKX_BACKENDS_CPU_RUNTIME_CALL_THUNK_H_

#include <memory>

#include "absl/status/statusor.h"

#include "zkx/backends/cpu/runtime/thunk.h"
#include "zkx/backends/cpu/runtime/thunk_executor.h"

namespace zkx::cpu {

// A thunk constructed from a call instruction that simply calls a thunk
// sequence emitted from the called computation.
class CallThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<CallThunk>> Create(
      Info info, ThunkSequence called_sequence);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;
  ResourceUses resource_uses() const final;

  const ThunkExecutor& called_executor() const { return called_executor_; }

 private:
  CallThunk(Info info, ThunkExecutor called_executor);

  ThunkExecutor called_executor_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_RUNTIME_CALL_THUNK_H_
