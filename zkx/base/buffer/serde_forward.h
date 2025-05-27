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
    T,
    std::void_t<
        decltype(Serde<T>::WriteTo(std::declval<const T&>(),
                                   std::declval<Buffer*>())),
        decltype(Serde<T>::ReadFrom(
            std::declval<const ReadOnlyBuffer&>(),
            std::declval<std::conditional_t<is_bounded_array_v<T>, T, T*>>())),
        decltype(Serde<T>::EstimateSize(std::declval<const T&>()))>>
    : std::true_type {};

template <typename... Args>
size_t EstimateSize(const Args&... args) {
  return (... + Serde<Args>::EstimateSize(args));
}

}  // namespace zkx::base

#endif  // ZKX_BASE_BUFFER_SERDE_FORWARD_H_
