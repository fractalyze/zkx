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

#ifndef ZKX_BASE_BUFFER_ENDIAN_H_
#define ZKX_BASE_BUFFER_ENDIAN_H_

#include <ostream>
#include <string_view>

namespace zkx::base {

enum class Endian {
  kNative,
  kBig,
  kLittle,
};

std::string_view EndianToString(Endian endian);

std::ostream& operator<<(std::ostream& os, Endian endian);

}  // namespace zkx::base

#endif  // ZKX_BASE_BUFFER_ENDIAN_H_
