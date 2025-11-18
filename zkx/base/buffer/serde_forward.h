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

#ifndef ZKX_BASE_BUFFER_SERDE_FORWARD_H_
#define ZKX_BASE_BUFFER_SERDE_FORWARD_H_

#include <type_traits>

#include "zkx/base/types/cxx20_is_bounded_array.h"

namespace zkx::base {

class Buffer;
class ReadOnlyBuffer;

// NOTE: Do not implement for builtin serde.
// See zkx/base/buffer/read_only_buffer.h
template <typename T, typename SFINAE = void>
class Serde;

template <typename, typename = void>
struct IsSerde : std::false_type {};

template <typename T>
struct IsSerde<
    T, std::void_t<
           decltype(Serde<T>::WriteTo(std::declval<const T&>(),
                                      std::declval<Buffer*>(),
                                      std::declval<Endian>())),
           decltype(Serde<T>::ReadFrom(
               std::declval<const ReadOnlyBuffer&>(),
               std::declval<std::conditional_t<is_bounded_array_v<T>, T, T*>>(),
               std::declval<Endian>())),
           decltype(Serde<T>::EstimateSize(std::declval<const T&>()))>>
    : std::true_type {};

template <typename... Args>
size_t EstimateSize(const Args&... args) {
  return (... + Serde<Args>::EstimateSize(args));
}

}  // namespace zkx::base

#endif  // ZKX_BASE_BUFFER_SERDE_FORWARD_H_
