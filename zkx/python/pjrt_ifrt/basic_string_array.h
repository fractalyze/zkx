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

#ifndef ZKX_PYTHON_PJRT_IFRT_BASIC_STRING_ARRAY_H_
#define ZKX_PYTHON_PJRT_IFRT_BASIC_STRING_ARRAY_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include "xla/tsl/concurrency/ref_count.h"
#include "zkx/pjrt/pjrt_layout.h"
#include "zkx/python/ifrt/array.h"
#include "zkx/python/ifrt/device.h"
#include "zkx/python/ifrt/device_list.h"
#include "zkx/python/ifrt/dtype.h"
#include "zkx/python/ifrt/future.h"
#include "zkx/python/ifrt/memory.h"
#include "zkx/python/ifrt/shape.h"
#include "zkx/python/ifrt/sharding.h"
#include "zkx/python/ifrt/user_context.h"

namespace zkx::ifrt {

// `BasicStringArray` implements an `ifrt::Array` by wrapping a local (aka host)
// string buffer. This object is expected to live exclusively in the IFRT layer,
// and thus is not specific to any particular backend. However, it is currently
// located in the pjrt_ifrt directory because we expect the main use of this
// class is to implement IO Callable support in pjrt_ifrt.
class BasicStringArray final
    : public llvm::RTTIExtends<BasicStringArray, Array> {
 public:
  // Must be in dense major to minor order.
  using Buffer = absl::Span<const absl::Cord>;

  // One Buffer per shard.
  static constexpr int kBuffersInlineSize = 1;
  using Buffers = absl::InlinedVector<Buffer, kBuffersInlineSize>;

  // Called when this object is done with the string buffer provided at the
  // construction time.
  using OnDoneWithBuffer = std::function<void()>;

  // General array construction. The `buffers` and their elements
  // (absl::Cords) must live until the `on_done_with_buffer` is called.
  // The number and order of buffers must match the number and order of devices
  // in `sharding`.
  static absl::StatusOr<tsl::RCReference<BasicStringArray>> Create(
      Client* client, Shape shape, ShardingRef sharding,
      Future<Buffers> buffers, OnDoneWithBuffer on_done_with_buffer);

  ~BasicStringArray() override;

  absl::StatusOr<ArrayRef> FullyReplicatedShard(
      ArrayCopySemantics semantics) override;

  // ifrt::Array API

  Client* client() const override { return client_; }

  DType dtype() const override { return DType(DType::kString); }

  const Shape& shape() const override { return shape_; }

  const Sharding& sharding() const override { return *sharding_; }

  ShardingRef shared_ptr_sharding() const override { return sharding_; }

  absl::StatusOr<std::shared_ptr<const zkx::PjRtLayout>> pjrt_layout()
      const override;

  UserContextRef user_context() const override { return user_context_; }

  absl::StatusOr<std::vector<ArrayRef>> DisassembleIntoSingleDeviceArrays(
      ArrayCopySemantics array_copy_semantics,
      SingleDeviceShardSemantics single_device_shard_semantics) override;

  ABSL_MUST_USE_RESULT
  Future<> CopyToHostBuffer(
      void* data, std::optional<absl::Span<const int64_t>> byte_strides,
      ArrayCopySemantics semantics) override;

  absl::StatusOr<ArrayRef> Copy(std::optional<DeviceListRef> devices,
                                std::optional<MemoryKind> memory_kind,
                                ArrayCopySemantics semantics);

  Future<> GetReadyFuture() const override;

  Future<> Delete() override;
  bool IsDeleted() const override;

  std::string DebugString() const override;

  // Methods specific to this Array variant (i.e., not from `ifrt::Array`).

  // Returns a future holding the string buffers underlying this array. Valid
  // only while this Array object is alive.
  Future<Buffers> buffers() const {
    return buffers_;  // Future copying is not considered expensive.
  }

  static char ID;

 private:
  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  BasicStringArray(Client* client, Shape shape, ShardingRef sharding,
                   Future<Buffers> buffers, Future<> ready_future,
                   OnDoneWithBuffer on_done_with_buffer);

  // Internal implementation of delete.
  void DeleteInternal() ABSL_LOCKS_EXCLUDED(mu_);

  Client* client_;  // not owned
  Shape shape_;
  ShardingRef sharding_;
  const UserContextRef user_context_;
  Future<Buffers> buffers_;
  Future<> ready_future_;

  mutable absl::Mutex mu_;
  OnDoneWithBuffer on_done_with_buffer_ ABSL_GUARDED_BY(mu_);
  bool is_deleted_ ABSL_GUARDED_BY(mu_) = false;
};

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_PJRT_IFRT_BASIC_STRING_ARRAY_H_
