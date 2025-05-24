#ifndef ZKX_BASE_TEMPLATE_UTIL_H_
#define ZKX_BASE_TEMPLATE_UTIL_H_

#include <type_traits>
#include <utility>

namespace zkx::base::internal {

template <typename, typename = std::void_t<>>
struct has_resize : std::false_type {};

template <typename T>
struct has_resize<T, std::void_t<decltype(std::declval<T&>().resize(
                         std::declval<std::size_t>()))>> : std::true_type {};

template <typename T>
constexpr bool has_resize_v = has_resize<T>::value;

}  // namespace zkx::base::internal

#endif  // ZKX_BASE_TEMPLATE_UTIL_H_
