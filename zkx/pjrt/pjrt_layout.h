/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

#ifndef ZKX_PJRT_PJRT_LAYOUT_H_
#define ZKX_PJRT_PJRT_LAYOUT_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/layout.h"

namespace zkx {

// Represents the memory layout of a PjRtBuffer.
class PjRtLayout {
 public:
  explicit PjRtLayout(Layout layout) : zkx_layout_(std::move(layout)) {
    // Strip memory space and set it to the default. PJRT tracks memory space
    // separately from layout.
    zkx_layout_.set_memory_space(Layout::kDefaultMemorySpace);
  }

  PjRtLayout(PjRtLayout& other) = delete;
  PjRtLayout& operator=(const PjRtLayout& other) = delete;

  static absl::StatusOr<std::shared_ptr<const PjRtLayout>> Deserialize(
      std::string_view serialized) {
    TF_ASSIGN_OR_RETURN(Layout layout, ParseLayout(serialized));
    return std::make_shared<PjRtLayout>(std::move(layout));
  }

  const Layout& zkx_layout() const { return zkx_layout_; }

  // Returns the serialized layout as a string.
  std::string Serialize() const { return zkx_layout_.ToString(); }

  // Human-readable string for error messages, user introspection, etc.
  std::string ToString() const { return zkx_layout_.ToString(); }

  bool operator==(const PjRtLayout& other) const {
    return zkx_layout_ == other.zkx_layout_;
  }

  template <typename H>
  friend H AbslHashValue(H state, const PjRtLayout& layout) {
    return H::combine(std::move(state), layout.zkx_layout_);
  }

 private:
  Layout zkx_layout_;
};

}  // namespace zkx

#endif  // ZKX_PJRT_PJRT_LAYOUT_H_
