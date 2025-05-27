#ifndef ZKX_BASE_BUFFER_SERDE_H_
#define ZKX_BASE_BUFFER_SERDE_H_

#include <array>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"

#include "zkx/base/buffer/buffer.h"
#include "zkx/base/buffer/serde_forward.h"

namespace zkx::base {

// NOTE(chokobole): `WriteTo()` and `ReadFrom()` are intentionally omitted.
// This specialization only provides `EstimateSize()` to support usage by
// other `Serde<T>` implementations (e.g., Serde<int>::EstimateSize(1)).
template <typename T>
class Serde<T, std::enable_if_t<internal::IsBuiltinSerde<T>::value>> {
 public:
  static size_t EstimateSize(const T& value) { return sizeof(T); }
};

template <typename T>
class Serde<T, std::enable_if_t<std::is_enum_v<T>>> {
 public:
  static absl::Status WriteTo(const T& value, Buffer* buffer) {
    return buffer->Write(static_cast<std::underlying_type_t<T>>(value));
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer, T* value) {
    std::underlying_type_t<T> underlying_value;
    TF_RETURN_IF_ERROR(buffer.Read(&underlying_value));
    *value = static_cast<T>(underlying_value);
    return absl::OkStatus();
  }

  static size_t EstimateSize(T value) { return sizeof(T); }
};

template <typename CharTy>
class Serde<std::basic_string_view<CharTy>> {
 public:
  static absl::Status WriteTo(const std::basic_string_view<CharTy>& value,
                              Buffer* buffer) {
    TF_RETURN_IF_ERROR(buffer->Write(value.size()));
    return buffer->Write(reinterpret_cast<const uint8_t*>(value.data()),
                         value.size());
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               std::basic_string_view<CharTy>* value) {
    ABSL_UNREACHABLE();
    return absl::InternalError(
        "std::basic_string_view<CharTy> is not deserializable");
  }

  static size_t EstimateSize(const std::basic_string_view<CharTy>& value) {
    return sizeof(size_t) + value.size();
  }
};

template <typename CharTy>
class Serde<std::basic_string<CharTy>> {
 public:
  static absl::Status WriteTo(const std::basic_string<CharTy>& value,
                              Buffer* buffer) {
    TF_RETURN_IF_ERROR(buffer->Write(value.size()));
    return buffer->Write(reinterpret_cast<const uint8_t*>(value.data()),
                         value.size());
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               std::basic_string<CharTy>* value) {
    size_t size;
    TF_RETURN_IF_ERROR(buffer.Read(&size));
    value->resize(size);
    return buffer.Read(reinterpret_cast<uint8_t*>(value->data()), size);
  }

  static size_t EstimateSize(const std::basic_string<CharTy>& value) {
    return sizeof(size_t) + value.size();
  }
};

template <typename CharTy>
class Serde<const CharTy*, std::enable_if_t<std::is_same_v<CharTy, char> ||
                                            std::is_same_v<CharTy, wchar_t> ||
                                            std::is_same_v<CharTy, char16_t> ||
                                            std::is_same_v<CharTy, char32_t>>> {
 public:
  static absl::Status WriteTo(const CharTy* value, Buffer* buffer) {
    size_t length = std::char_traits<CharTy>::length(value);
    TF_RETURN_IF_ERROR(buffer->Write(length));
    return buffer->Write(reinterpret_cast<const uint8_t*>(value),
                         length * sizeof(CharTy));
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               const CharTy** value) {
    ABSL_UNREACHABLE();
    return absl::InternalError("const CharTy* is not deserializable");
  }

  static size_t EstimateSize(const CharTy* value) {
    return sizeof(size_t) +
           std::char_traits<CharTy>::length(value) * sizeof(CharTy);
  }
};

template <typename T, size_t N>
class Serde<T[N]> {
 public:
  static absl::Status WriteTo(const T* values, Buffer* buffer) {
    for (size_t i = 0; i < N; ++i) {
      TF_RETURN_IF_ERROR(buffer->Write(values[i]));
    }
    return absl::OkStatus();
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer, T* values) {
    for (size_t i = 0; i < N; ++i) {
      TF_RETURN_IF_ERROR(buffer.Read(&values[i]));
    }
    return absl::OkStatus();
  }

  static size_t EstimateSize(const T* values) {
    return std::accumulate(values, &values[N], 0,
                           [](size_t total, const T& value) {
                             return total + base::EstimateSize(value);
                           });
  }
};

template <typename T>
class Serde<std::vector<T>> {
 public:
  static absl::Status WriteTo(const std::vector<T>& values, Buffer* buffer) {
    TF_RETURN_IF_ERROR(buffer->Write(values.size()));
    for (const T& value : values) {
      TF_RETURN_IF_ERROR(buffer->Write(value));
    }
    return absl::OkStatus();
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               std::vector<T>* values) {
    size_t size;
    TF_RETURN_IF_ERROR(buffer.Read(&size));
    values->resize(size);
    for (T& value : (*values)) {
      TF_RETURN_IF_ERROR(buffer.Read(&value));
    }
    return absl::OkStatus();
  }

  static size_t EstimateSize(const std::vector<T>& values) {
    return std::accumulate(values.begin(), values.end(), sizeof(size_t),
                           [](size_t total, const T& value) {
                             return total + base::EstimateSize(value);
                           });
  }
};

template <typename T, size_t N>
class Serde<std::array<T, N>> {
 public:
  static absl::Status WriteTo(const std::array<T, N>& values, Buffer* buffer) {
    for (const T& value : values) {
      TF_RETURN_IF_ERROR(buffer->Write(value));
    }
    return absl::OkStatus();
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               std::array<T, N>* values) {
    for (T& value : (*values)) {
      TF_RETURN_IF_ERROR(buffer.Read(&value));
    }
    return absl::OkStatus();
  }

  static size_t EstimateSize(const std::array<T, N>& values) {
    return std::accumulate(values.begin(), values.end(), 0,
                           [](size_t total, const T& value) {
                             return total + base::EstimateSize(value);
                           });
  }
};

template <typename T>
class Serde<absl::Span<T>> {
 public:
  static absl::Status WriteTo(absl::Span<T> values, Buffer* buffer) {
    TF_RETURN_IF_ERROR(buffer->Write(values.size()));
    for (const T& value : values) {
      TF_RETURN_IF_ERROR(buffer->Write(value));
    }
    return absl::OkStatus();
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               absl::Span<T>* values) {
    ABSL_UNREACHABLE();
    return absl::InternalError("absl::Span<T> is not deserializable");
  }

  static size_t EstimateSize(absl::Span<T> values) {
    return std::accumulate(values.begin(), values.end(), sizeof(size_t),
                           [](size_t total, const T& value) {
                             return total + base::EstimateSize(value);
                           });
  }
};

template <typename... Ts>
class Serde<std::tuple<Ts...>> {
 public:
  static absl::Status WriteTo(const std::tuple<Ts...>& values, Buffer* buffer) {
    return std::apply(
        [buffer](const auto&... values) {
          absl::Status status = absl::OkStatus();
          (void)((status = buffer->Write(values), status.ok()) && ...);
          return status;
        },
        values);
  }

  static absl::Status ReadFrom(const ReadOnlyBuffer& buffer,
                               std::tuple<Ts...>* values) {
    return std::apply(
        [&buffer](auto&... values) {
          absl::Status status = absl::OkStatus();
          (void)((status = buffer.Read(&values), status.ok()) && ...);
          return status;
        },
        *values);
  }

  static size_t EstimateSize(const std::tuple<Ts...>& values) {
    return std::apply(
        [](const auto&... values) {
          return (base::EstimateSize(values) + ...);
        },
        values);
  }
};

}  // namespace zkx::base

#endif  // ZKX_BASE_BUFFER_SERDE_H_
