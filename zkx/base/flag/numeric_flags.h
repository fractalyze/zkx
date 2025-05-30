#ifndef ZKX_BASE_FLAG_NUMERIC_FLAGS_H_
#define ZKX_BASE_FLAG_NUMERIC_FLAGS_H_

#include "xla/tsl/platform/errors.h"
#include "zkx/base/flag/flag_value_traits.h"

namespace zkx::base {

template <typename T, typename std::enable_if_t<std::is_signed_v<T>>* = nullptr>
absl::Status ParsePositiveValue(std::string_view arg, T* value) {
  T n;
  TF_RETURN_IF_ERROR(FlagValueTraits<T>::ParseValue(arg, &n));
  if (n > 0) {
    *value = n;
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::Substitute("$0 is not positive", arg));
}

}  // namespace zkx::base

#endif  // ZKX_BASE_FLAG_NUMERIC_FLAGS_H_
