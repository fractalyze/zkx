/* Copyright 2022 The OpenXLA Authors.
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

#ifndef ZKX_PYTHON_PJRT_IFRT_PJRT_ARRAY_H_
#define ZKX_PYTHON_PJRT_IFRT_PJRT_ARRAY_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include "xla/tsl/concurrency/ref_count.h"
#include "zkx/pjrt/pjrt_client.h"
#include "zkx/pjrt/pjrt_layout.h"
#include "zkx/python/ifrt/array.h"
#include "zkx/python/ifrt/device_list.h"
#include "zkx/python/ifrt/dtype.h"
#include "zkx/python/ifrt/future.h"
#include "zkx/python/ifrt/memory.h"
#include "zkx/python/ifrt/shape.h"
#include "zkx/python/ifrt/sharding.h"
#include "zkx/python/ifrt/user_context.h"
#include "zkx/python/pjrt_ifrt/pjrt_client.h"

namespace zkx::ifrt {

// Creates IFRT `MemoryKind` from a ZKX `PjRtBuffer`.
MemoryKind MakeMemoryKindFromPjRtBuffer(PjRtBuffer* pjrt_buffer);

// PjRt-compatible `Array` interface that wraps a list of `ZKX::PjRtBuffer`s.
class PjRtCompatibleArray
    : public llvm::RTTIExtends<PjRtCompatibleArray, Array> {
 public:
  // APIs that allow direct access to `PjRtBuffer` for PjRt-only operations.
  virtual absl::Span<const std::shared_ptr<PjRtBuffer>> pjrt_buffers() = 0;
  virtual absl::StatusOr<absl::Span<std::shared_ptr<PjRtBuffer>>>
  mutable_pjrt_buffers() = 0;

  static char ID;
};

// `Array` implementation that wraps a list of `PjRtBuffer`s.
class PjRtArray final
    : public llvm::RTTIExtends<PjRtArray, PjRtCompatibleArray> {
 public:
  static constexpr int kPjRtBufferInlineSize = 1;
  using PjRtBuffers =
      absl::InlinedVector<std::shared_ptr<PjRtBuffer>, kPjRtBufferInlineSize>;

  // General array construction (with static shape). `pjrt_buffers` may be
  // empty. `layout == nullptr` indicates a default layout.
  static absl::StatusOr<tsl::RCReference<PjRtArray>> Create(
      PjRtCompatibleClient* client, DType dtype, Shape shape,
      ShardingRef sharding, PjRtBuffers pjrt_buffers,
      std::shared_ptr<const zkx::PjRtLayout> layout);

  // General array construction (with dynamic shape). `pjrt_buffers` may be
  // empty. `layout == nullptr` indicates a default layout.
  static absl::StatusOr<tsl::RCReference<PjRtArray>> Create(
      PjRtCompatibleClient* client, DType dtype, DynamicShape dynamic_shape,
      ShardingRef sharding, PjRtBuffers pjrt_buffers,
      std::shared_ptr<const zkx::PjRtLayout> layout);

  // Shorthand for a single-shard array construction.
  // See `PjRtCompatibleClient::CreatePjRtArray()` for the meaning of
  // `has_custom_layout`.
  static absl::StatusOr<tsl::RCReference<PjRtArray>> Create(
      PjRtCompatibleClient* client, std::shared_ptr<PjRtBuffer> pjrt_buffer,
      bool has_custom_layout);

  // Shorthand for a multi-shard array construction using ConcreteSharding.
  // `pjrt_buffers` must be non-empty.
  // See `PjRtCompatibleClient::CreatePjRtArray()` for the meaning of
  // `has_custom_layout`.
  // TODO(hyeontaek): Remove this once IFRT Sharding and JAX Sharding is unified
  // so that ConcreteSharding can be replaced with a real Sharding.
  static absl::StatusOr<tsl::RCReference<PjRtArray>> Create(
      PjRtCompatibleClient* client, Shape shape, PjRtBuffers pjrt_buffers,
      bool has_custom_layout);

  // Shorthand for a multi-shard array construction using ConcreteSharding with
  // DynamicShape. `pjrt_buffers` must be non-empty.
  // See `PjRtCompatibleClient::CreatePjRtArray()` for the meaning of
  // `has_custom_layout`.
  static absl::StatusOr<tsl::RCReference<PjRtArray>> Create(
      PjRtCompatibleClient* client, DynamicShape dynamic_shape,
      PjRtBuffers pjrt_buffers, bool has_custom_layout);

  // PjRtCompatibleArray implementation.

  absl::Span<const std::shared_ptr<PjRtBuffer>> pjrt_buffers() override {
    DCHECK(this);
    return pjrt_buffers_;
  }
  absl::StatusOr<absl::Span<std::shared_ptr<PjRtBuffer>>> mutable_pjrt_buffers()
      override {
    DCHECK(this);
    return absl::MakeSpan(pjrt_buffers_);
  }

  absl::StatusOr<ArrayRef> FullyReplicatedShard(
      ArrayCopySemantics semantics) override;

  // Array implementation.

  ~PjRtArray() override = default;

  PjRtCompatibleClient* client() const override {
    DCHECK(this);
    return client_;
  }

  DType dtype() const override {
    DCHECK(this);
    return dtype_;
  }

  bool has_dynamic_shape() const {
    DCHECK(this);
    return std::holds_alternative<DynamicShape>(shape_);
  }

  bool has_static_shape() const {
    DCHECK(this);
    return std::holds_alternative<Shape>(shape_);
  }

  const Shape& shape() const override {
    DCHECK(has_static_shape());
    return std::get<Shape>(shape_);
  }

  const DynamicShape& dynamic_shape() const {
    DCHECK(has_dynamic_shape());
    return std::get<DynamicShape>(shape_);
  }

  const Sharding& sharding() const override {
    DCHECK(this);
    return *sharding_;
  }
  ShardingRef shared_ptr_sharding() const override {
    DCHECK(this);
    return sharding_;
  }

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

  std::shared_ptr<PjRtBuffer> GetPjRtBuffer(ArrayCopySemantics semantics,
                                            int index) const;

  Future<> Delete() override;
  bool IsDeleted() const override;

  std::string DebugString() const override;

  static char ID;

 private:
  PjRtArray(PjRtCompatibleClient* client, DType dtype, Shape shape,
            ShardingRef sharding, PjRtBuffers pjrt_buffers,
            std::shared_ptr<const zkx::PjRtLayout> layout);

  PjRtArray(PjRtCompatibleClient* client, DType dtype,
            DynamicShape dynamic_shape, ShardingRef sharding,
            PjRtBuffers pjrt_buffers,
            std::shared_ptr<const zkx::PjRtLayout> layout);

  template <typename T, typename... Args>
  friend tsl::RCReference<T> tsl::MakeRef(Args&&... args);

  PjRtCompatibleClient* client_;  // not owned
  DType dtype_;
  std::variant<Shape, DynamicShape> shape_;
  ShardingRef sharding_;
  PjRtBuffers pjrt_buffers_;
  std::shared_ptr<const zkx::PjRtLayout> layout_;
  const UserContextRef user_context_;
  bool is_deleted_ = false;
};

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_PJRT_IFRT_PJRT_ARRAY_H_
