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

#include "zkx/base/buffer/buffer.h"

#include <string.h>

#include "absl/base/internal/endian.h"

namespace zkx::base {

absl::Status Buffer::WriteAt(size_t buffer_offset, const uint8_t* ptr,
                             size_t size) {
  // TODO(chokobole): add overflow check
  size_t size_needed = buffer_offset + size;
  if (buffer_offset + size > buffer_len_) {
    TF_RETURN_IF_ERROR(Grow(size_needed));
  }
  memcpy(reinterpret_cast<char*>(buffer_) + buffer_offset, ptr, size);
  buffer_offset_ = buffer_offset + size;
  return absl::OkStatus();
}

#define WRITE_BE_AT(bytes, bits, type)                                       \
  absl::Status Buffer::Write##bits##BEAt(size_t buffer_offset, type value) { \
    /* TODO(chokobole): add overflow check */                                \
    size_t size_needed = buffer_offset + bytes;                              \
    if (size_needed > buffer_len_) {                                         \
      TF_RETURN_IF_ERROR(Grow(size_needed));                                 \
    }                                                                        \
    char* buffer = reinterpret_cast<char*>(buffer_);                         \
    absl::big_endian::Store##bits(&buffer[buffer_offset], value);            \
    buffer_offset_ = buffer_offset + bytes;                                  \
    return absl::OkStatus();                                                 \
  }

WRITE_BE_AT(2, 16, uint16_t)
WRITE_BE_AT(4, 32, uint32_t)
WRITE_BE_AT(8, 64, uint64_t)

#undef WRITE_BE_AT

#define WRITE_LE_AT(bytes, bits, type)                                       \
  absl::Status Buffer::Write##bits##LEAt(size_t buffer_offset, type value) { \
    /* TODO(chokobole): add overflow check */                                \
    size_t size_needed = buffer_offset + bytes;                              \
    if (size_needed > buffer_len_) {                                         \
      TF_RETURN_IF_ERROR(Grow(size_needed));                                 \
    }                                                                        \
    char* buffer = reinterpret_cast<char*>(buffer_);                         \
    absl::little_endian::Store##bits(&buffer[buffer_offset], value);         \
    buffer_offset_ = buffer_offset + bytes;                                  \
    return absl::OkStatus();                                                 \
  }

WRITE_LE_AT(2, 16, uint16_t)
WRITE_LE_AT(4, 32, uint32_t)
WRITE_LE_AT(8, 64, uint64_t)

#undef WRITE_LE_AT

}  // namespace zkx::base
