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

#ifndef ZKX_BASE_BUFFER_ENDIAN_AUTO_RESET_H_
#define ZKX_BASE_BUFFER_ENDIAN_AUTO_RESET_H_

#include "zkx/base/buffer/read_only_buffer.h"

namespace zkx::base {

struct EndianAutoReset {
  explicit EndianAutoReset(const base::ReadOnlyBuffer& buffer,
                           base::Endian endian)
      : buffer(buffer), old_endian(buffer.endian()) {
    buffer.set_endian(endian);
  }
  ~EndianAutoReset() { buffer.set_endian(old_endian); }

  const base::ReadOnlyBuffer& buffer;
  base::Endian old_endian;
};

}  // namespace zkx::base

#endif  // ZKX_BASE_BUFFER_ENDIAN_AUTO_RESET_H_
