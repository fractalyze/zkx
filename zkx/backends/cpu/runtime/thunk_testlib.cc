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

#include "zkx/backends/cpu/runtime/thunk_testlib.h"

#include <algorithm>

#include "xla/tsl/platform/statusor.h"
#include "zkx/stream_executor/device_memory.h"

namespace zkx::cpu {

BufferAllocation CreateBufferAllocation(size_t index, const Literal& literal) {
  size_t size_in_bytes = literal.size_bytes();
  return BufferAllocation(index, size_in_bytes, 0);
}

BufferAllocation::Slice CreateBufferAllocationSlice(
    const BufferAllocation& allocation) {
  return CreateBufferAllocationSlice(allocation, 0, allocation.size());
}

BufferAllocation::Slice CreateBufferAllocationSlice(
    const BufferAllocation& allocation, int64_t offset, int64_t size) {
  return BufferAllocation::Slice(&allocation, offset, size);
}

BufferAllocations CreateBufferAllocations(absl::Span<Literal*> literals) {
  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(literals.size());

  for (auto* literal : literals) {
    size_t size_in_bytes = literal->size_bytes();
    buffers.emplace_back(literal->untyped_data(), size_in_bytes);
  }

  return BufferAllocations(buffers);
}

namespace {

// We use a global static variable to simulate a shared resource. We check that
// thunk executor correctly orders access to this resource by running the test
// with a thread sanitizer and checking that there are no data races.
int64_t shared_resource;

}  // namespace

void InitSharedResource() { shared_resource = 0; }

int64_t GetSharedResource() { return shared_resource; }

absl::Status AddI32Thunk::Execute(const BufferAllocations* allocations,
                                  BufferAllocation::Slice src_slice,
                                  BufferAllocation::Slice dst_slice) {
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase src,
                      allocations->GetDeviceAddress(src_slice));

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase dst,
                      allocations->GetDeviceAddress(dst_slice));

  CHECK_EQ(src.size() % sizeof(int32_t), 0);
  CHECK_EQ(dst.size() % sizeof(int32_t), 0);

  int32_t* src_ptr = static_cast<int32_t*>(src.opaque());
  int32_t* dst_ptr = static_cast<int32_t*>(dst.opaque());
  size_t len = std::min(src.size(), dst.size()) / sizeof(int32_t);

  for (int j = 0; j < len; ++j) dst_ptr[j] += src_ptr[j];

  return absl::OkStatus();
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> AddI32Thunk::Execute(
    const ExecuteParams& params) {
  if (trace_) trace_->push_back(info().op_name);

  auto execute = [&]() -> absl::Status {
    CHECK_EQ(srcs_.size(), dsts_.size());
    for (int i = 0; i < srcs_.size(); ++i) {
      TF_RETURN_IF_ERROR(
          Execute(params.buffer_allocations, srcs_.at(i), dsts_.at(i)));
    }
    return absl::OkStatus();
  };

  // Offload the execution to the intra-op thread pool.
  if (params.intra_op_threadpool) {
    auto event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();
    params.intra_op_threadpool->getPool()->Schedule([&, event, execute] {
      if (use_shared_resource_) {
        shared_resource++;
      }

      if (inject_error_) {
        event.SetError(absl::InternalError("Injected error"));
      } else {
        CHECK_OK(execute());
        event.SetStateConcrete();
      }
    });
    return event;
  }

  if (use_shared_resource_) {
    shared_resource++;
  }

  if (inject_error_) {
    return tsl::MakeErrorAsyncValueRef(absl::InternalError("Injected error"));
  }

  TF_RETURN_IF_ERROR(execute());
  return Thunk::OkExecuteEvent();
}

AddI32Thunk::BufferUses AddI32Thunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const auto& src : srcs_) buffer_uses.push_back(BufferUse::Read(src));
  for (const auto& dst : dsts_) buffer_uses.push_back(BufferUse::Write(dst));
  return buffer_uses;
}

AddI32Thunk::ResourceUses AddI32Thunk::resource_uses() const {
  static std::shared_ptr<Resource>* shared_resource =
      new std::shared_ptr<Resource>(Resource::Create(Resource::kToken));

  return use_shared_resource_
             ? ResourceUses{ResourceUse::Write(*shared_resource)}
             : ResourceUses{};
}

}  // namespace zkx::cpu
