#ifndef ZKX_MATH_BASE_FIELD_H_
#define ZKX_MATH_BASE_FIELD_H_

namespace zkx::math {

template <typename T, typename SFINAE = void>
struct IsFieldImpl {
  constexpr static bool value = false;
};

template <typename T>
constexpr bool IsField = IsFieldImpl<T>::value;

}  // namespace zkx::math

#endif  // ZKX_MATH_BASE_FIELD_H_
