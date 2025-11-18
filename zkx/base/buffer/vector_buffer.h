/* Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_BASE_BUFFER_VECTOR_BUFFER_H_
#define ZKX_BASE_BUFFER_VECTOR_BUFFER_H_

#include <utility>
#include <vector>

#include "zkx/base/buffer/buffer.h"

namespace zkx::base {

template <typename T>
class VectorBuffer : public Buffer {
 public:
  static_assert(sizeof(T) == 1);

  VectorBuffer() = default;
  explicit VectorBuffer(const std::vector<T>& owned_buffer)
      : owned_buffer_(owned_buffer) {
    UpdateBuffer();
  }
  explicit VectorBuffer(std::vector<T>&& owned_buffer)
      : owned_buffer_(std::move(owned_buffer)) {
    UpdateBuffer();
  }
  VectorBuffer(const VectorBuffer& other) = delete;
  VectorBuffer& operator=(const VectorBuffer& other) = delete;
  VectorBuffer(VectorBuffer&& other)
      : Buffer(std::move(other)),
        owned_buffer_(std::move(other.owned_buffer_)) {}
  VectorBuffer& operator=(VectorBuffer&& other) {
    Buffer::operator=(std::move(other));
    owned_buffer_ = std::move(other.owned_buffer_);
    return *this;
  }
  ~VectorBuffer() override = default;

  const std::vector<T>& owned_buffer() const { return owned_buffer_; }

  std::vector<T>&& TakeOwnedBuffer() && { return std::move(owned_buffer_); }

  absl::Status Grow(size_t size) override {
    owned_buffer_.resize(size);
    UpdateBuffer();
    return absl::OkStatus();
  }

 protected:
  void UpdateBuffer() {
    buffer_ = owned_buffer_.data();
    buffer_len_ = owned_buffer_.size();
  }

  std::vector<T> owned_buffer_;
};

using CharVectorBuffer = VectorBuffer<char>;
using Uint8VectorBuffer = VectorBuffer<uint8_t>;

extern template class VectorBuffer<char>;
extern template class VectorBuffer<uint8_t>;

}  // namespace zkx::base

#endif  // ZKX_BASE_BUFFER_VECTOR_BUFFER_H_
