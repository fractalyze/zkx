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

#include "zkx/backends/cpu/runtime/thunk.h"

namespace zkx::cpu {

std::string_view Thunk::KindToString(Kind kind) {
  switch (kind) {
    case Kind::kAllGather:
      return "all-gather";
    case Kind::kAllReduce:
      return "all-reduce";
    case Kind::kAllToAll:
      return "all-to-all";
    case Kind::kCollectivePermute:
      return "collective-permute";
    case Kind::kKernel:
      return "kernel";
    case Kind::kReduceScatter:
      return "reduce-scatter";
    case Kind::kUnknown:
      return "unknown";
  }
}

Thunk::Thunk(Kind kind, Info info)
    : kind_(kind),
      info_(std::move(info)),
      ok_event_(OkExecuteEventSingleton()) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> Thunk::OkExecuteEventSingleton() {
  static tsl::AsyncValueOwningRef<ExecuteEvent>* singleton = [] {
    auto* storage = new tsl::internal::AsyncValueStorage<ExecuteEvent>();
    return new tsl::AsyncValueOwningRef<ExecuteEvent>(
        tsl::MakeAvailableAsyncValueRef<ExecuteEvent>(*storage));
  }();
  return singleton->AsRef();
}

Thunk::ExecuteSession::ExecuteSession(int64_t max_workers,
                                      int64_t split_threshold)
    : lock_(std::make_shared<std::nullopt_t>(std::nullopt)),
      max_workers_(max_workers),
      split_threshold_(split_threshold) {}

ThunkSequence::ThunkSequence(std::unique_ptr<Thunk> thunk) {
  push_back(std::move(thunk));
}

void ThunkSequence::Append(ThunkSequence other) {
  reserve(size() + other.size());
  for (auto& thunk : other) {
    push_back(std::move(thunk));
  }
}

ThunkSequence::BufferUses ThunkSequence::buffer_uses() const {
  BufferUses buffer_uses;
  for (auto& thunk : *this) {
    BufferUses uses = thunk->buffer_uses();
    buffer_uses.insert(buffer_uses.end(), uses.begin(), uses.end());
  }
  return buffer_uses;
}

ThunkSequence::ResourceUses ThunkSequence::resource_uses() const {
  ResourceUses resource_uses;
  for (auto& thunk : *this) {
    ResourceUses uses = thunk->resource_uses();
    resource_uses.insert(resource_uses.end(), uses.begin(), uses.end());
  }
  return resource_uses;
}

}  // namespace zkx::cpu
