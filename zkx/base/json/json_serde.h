#ifndef ZKX_BASE_JSON_JSON_SERDE_H_
#define ZKX_BASE_JSON_JSON_SERDE_H_

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/numeric/bits.h"
#include "absl/strings/substitute.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"

#include "xla/tsl/platform/statusor.h"

namespace zkx::base {

std::string RapidJsonMismatchedTypeError(std::string_view key,
                                         std::string_view type,
                                         const rapidjson::Value& value);

template <typename T>
std::string RapidJsonOutOfRangeError(std::string_view key, T value) {
  return absl::Substitute("value($0) of \"$1\" is out of range", value, key);
}

template <typename T, typename SFINAE = void>
class JsonSerde;

template <>
class JsonSerde<bool> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(bool value, Allocator& allocator) {
    return rapidjson::Value(value);
  }

  static absl::StatusOr<bool> To(const rapidjson::Value& json_value,
                                 std::string_view key) {
    if (!json_value.IsBool()) {
      return absl::InvalidArgumentError(
          RapidJsonMismatchedTypeError(key, "bool", json_value));
    }
    return json_value.GetBool();
  }
};

template <typename T>
class JsonSerde<T,
                std::enable_if_t<std::is_integral<T>::value &&
                                 std::is_signed<T>::value && sizeof(T) == 8>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(T value, Allocator& allocator) {
    int64_t i64_value = absl::bit_cast<int64_t>(value);
    return rapidjson::Value(i64_value);
  }

  static absl::StatusOr<T> To(const rapidjson::Value& json_value,
                              std::string_view key) {
    if (!json_value.IsInt64() && !json_value.IsInt()) {
      return absl::InvalidArgumentError(
          RapidJsonMismatchedTypeError(key, "int64", json_value));
    }
    if (json_value.IsInt()) {
      return absl::bit_cast<T>(static_cast<int64_t>(json_value.GetInt()));
    } else {
      return absl::bit_cast<T>(json_value.GetInt64());
    }
  }
};

template <typename T>
class JsonSerde<T,
                std::enable_if_t<std::is_integral<T>::value &&
                                 !std::is_signed<T>::value && sizeof(T) == 8>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(T value, Allocator& allocator) {
    uint64_t u64_value = absl::bit_cast<uint64_t>(value);
    return rapidjson::Value(u64_value);
  }

  static absl::StatusOr<T> To(const rapidjson::Value& json_value,
                              std::string_view key) {
    if (!json_value.IsUint64() && !json_value.IsUint()) {
      return absl::InvalidArgumentError(
          RapidJsonMismatchedTypeError(key, "uint64", json_value));
    }
    if (json_value.IsUint()) {
      return absl::bit_cast<T>(static_cast<uint64_t>(json_value.GetUint()));
    } else {
      return absl::bit_cast<T>(json_value.GetUint64());
    }
  }
};

template <typename T>
class JsonSerde<
    T,
    std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value &&
                     sizeof(T) < 8>> {  // NOLINT(whitespace/operators)
 public:
  template <typename Allocator>
  static rapidjson::Value From(T value, Allocator& allocator) {
    static_assert(sizeof(T) <= sizeof(int));
    return rapidjson::Value(static_cast<int>(value));
  }

  static absl::StatusOr<T> To(const rapidjson::Value& json_value,
                              std::string_view key) {
    if (!json_value.IsInt()) {
      if (json_value.IsUint()) {
        return absl::OutOfRangeError(
            RapidJsonOutOfRangeError(key, json_value.GetUint()));
      } else if (json_value.IsUint64()) {
        return absl::OutOfRangeError(
            RapidJsonOutOfRangeError(key, json_value.GetUint64()));
      } else if (json_value.IsInt64()) {
        return absl::OutOfRangeError(
            RapidJsonOutOfRangeError(key, json_value.GetInt64()));
      }
      return absl::InvalidArgumentError(
          RapidJsonMismatchedTypeError(key, "int", json_value));
    }
    int64_t value = json_value.GetInt();
    if (value < int64_t{std::numeric_limits<T>::min()} ||
        value > int64_t{std::numeric_limits<T>::max()}) {
      return absl::OutOfRangeError(RapidJsonOutOfRangeError(key, value));
    }
    return static_cast<T>(value);
  }
};

template <typename T>
class JsonSerde<
    T,
    std::enable_if_t<std::is_integral<T>::value && !std::is_signed<T>::value &&
                     sizeof(T) < 8>> {  // NOLINT(whitespace/operators)
 public:
  template <typename Allocator>
  static rapidjson::Value From(T value, Allocator& allocator) {
    static_assert(sizeof(T) <= sizeof(unsigned));
    return rapidjson::Value(static_cast<unsigned>(value));
  }

  static absl::StatusOr<T> To(const rapidjson::Value& json_value,
                              std::string_view key) {
    if (!json_value.IsUint()) {
      if (json_value.IsInt()) {
        return absl::OutOfRangeError(
            RapidJsonOutOfRangeError(key, json_value.GetInt()));
      } else if (json_value.IsUint64()) {
        return absl::OutOfRangeError(
            RapidJsonOutOfRangeError(key, json_value.GetUint64()));
      } else if (json_value.IsInt64()) {
        return absl::OutOfRangeError(
            RapidJsonOutOfRangeError(key, json_value.GetInt64()));
      }
      return absl::InvalidArgumentError(
          RapidJsonMismatchedTypeError(key, "uint", json_value));
    }
    uint64_t value = json_value.GetUint();
    if (value > uint64_t{std::numeric_limits<T>::max()}) {
      return absl::OutOfRangeError(RapidJsonOutOfRangeError(key, value));
    }
    return static_cast<T>(value);
  }
};

template <>
class JsonSerde<float> {
 public:
  static bool s_allow_lossy_conversion;

  template <typename Allocator>
  static rapidjson::Value From(float value, Allocator& allocator) {
    return rapidjson::Value(value);
  }

  static absl::StatusOr<float> To(const rapidjson::Value& json_value,
                                  std::string_view key);
};

template <>
class JsonSerde<double> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(double value, Allocator& allocator) {
    return rapidjson::Value(value);
  }

  static absl::StatusOr<double> To(const rapidjson::Value& json_value,
                                   std::string_view key);
};

template <>
class JsonSerde<std::string> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const std::string& value, Allocator& allocator) {
    return rapidjson::Value(value.c_str(), value.length(), allocator);
  }

  static absl::StatusOr<std::string> To(const rapidjson::Value& json_value,
                                        std::string_view key) {
    if (!json_value.IsString()) {
      return absl::InvalidArgumentError(
          RapidJsonMismatchedTypeError(key, "string", json_value));
    }
    return json_value.GetString();
  }
};

template <>
class JsonSerde<std::string_view> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(std::string_view value, Allocator& allocator) {
    return rapidjson::Value(value.data(), value.length());
  }

  static absl::StatusOr<std::string_view> To(const rapidjson::Value& json_value,
                                             std::string_view key) {
    if (!json_value.IsString()) {
      return absl::InvalidArgumentError(
          RapidJsonMismatchedTypeError(key, "string_view", json_value));
    }
    return json_value.GetString();
  }
};

template <typename T>
class JsonSerde<T, std::enable_if_t<std::is_enum<T>::value>> {
 public:
  using U = std::underlying_type_t<T>;

  template <typename Allocator>
  static rapidjson::Value From(T value, Allocator& allocator) {
    return rapidjson::Value(static_cast<U>(value));
  }

  static absl::StatusOr<T> To(const rapidjson::Value& json_value,
                              std::string_view key) {
    TF_ASSIGN_OR_RETURN(U value, JsonSerde<U>::To(json_value, key));
    return static_cast<T>(value);
  }
};

template <typename T, size_t N>
class JsonSerde<std::array<T, N>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const std::array<T, N>& value,
                               Allocator& allocator) {
    rapidjson::Value array(rapidjson::kArrayType);
    for (size_t i = 0; i < N; ++i) {
      array.PushBack(JsonSerde<T>::From(value[i], allocator), allocator);
    }
    return array;
  }

  static absl::StatusOr<std::array<T, N>> To(const rapidjson::Value& json_value,
                                             std::string_view key) {
    if (!json_value.IsArray()) {
      return absl::InvalidArgumentError(
          RapidJsonMismatchedTypeError(key, "array", json_value));
    }
    if (N != json_value.Size()) {
      return absl::OutOfRangeError(absl::Substitute(
          "The length of json($0) is not $1", json_value.Size(), N));
    }
    std::array<T, N> value;
    for (size_t i = 0; i < N; ++i) {
      TF_ASSIGN_OR_RETURN(T v, JsonSerde<T>::To(json_value[i], key));
      value[i] = std::move(v);
    }
    return std::move(value);
  }
};

template <typename T>
class JsonSerde<std::vector<T>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const std::vector<T>& value,
                               Allocator& allocator) {
    rapidjson::Value array(rapidjson::kArrayType);
    for (size_t i = 0; i < value.size(); ++i) {
      array.PushBack(JsonSerde<T>::From(value[i], allocator), allocator);
    }
    return array;
  }

  static absl::StatusOr<std::vector<T>> To(const rapidjson::Value& json_value,
                                           std::string_view key) {
    if (!json_value.IsArray()) {
      return absl::InvalidArgumentError(
          RapidJsonMismatchedTypeError(key, "array", json_value));
    }
    std::vector<T> value;
    value.reserve(json_value.Size());
    for (auto it = json_value.Begin(); it != json_value.End(); ++it) {
      TF_ASSIGN_OR_RETURN(T v, JsonSerde<T>::To(*it, key));
      value.push_back(std::move(v));
    }
    return std::move(value);
  }
};

template <typename T>
class JsonSerde<std::optional<T>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const std::optional<T>& value,
                               Allocator& allocator) {
    if (value.has_value()) {
      return JsonSerde<T>::From(value.value(), allocator);
    } else {
      return rapidjson::Value();
    }
  }

  static absl::StatusOr<std::optional<T>> To(const rapidjson::Value& json_value,
                                             std::string_view key) {
    if (json_value.IsNull()) {
      return std::nullopt;
    }
    TF_ASSIGN_OR_RETURN(T value, JsonSerde<T>::To(json_value, key));
    return std::move(value);
  }
};

template <typename T, typename Allocator>
void AddJsonElement(rapidjson::Value& json_value, std::string_view key,
                    const T& value, Allocator& allocator) {
  json_value.AddMember(rapidjson::StringRef(key.data(), key.length()),
                       JsonSerde<T>::From(value, allocator), allocator);
}

template <typename T>
absl::StatusOr<T> ParseJsonElement(const rapidjson::Value& json_value,
                                   std::string_view key) {
  auto it = json_value.FindMember(key.data());
  if (it == json_value.MemberEnd()) {
    return absl::NotFoundError(
        absl::Substitute("\"$0\" key is not found", key));
  }
  return JsonSerde<T>::To(it->value, key);
}

}  // namespace zkx::base

namespace rapidjson {

template <typename OutputStream, typename SourceEncoding,
          typename TargetEncoding, typename StackAllocator>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           bool value) {
  writer.Bool(value);
  return writer;
}

template <typename OutputStream, typename SourceEncoding,
          typename TargetEncoding, typename StackAllocator>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           int64_t value) {
  writer.Int64(value);
  return writer;
}

template <typename OutputStream, typename SourceEncoding,
          typename TargetEncoding, typename StackAllocator>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           uint64_t value) {
  writer.Uint64(value);
  return writer;
}

template <
    typename OutputStream, typename SourceEncoding, typename TargetEncoding,
    typename StackAllocator, typename T,
    std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value &&
                     !std::is_same<T, int64_t>::value>* = nullptr>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           T value) {
  writer.Int(value);
  return writer;
}

template <
    typename OutputStream, typename SourceEncoding, typename TargetEncoding,
    typename StackAllocator, typename T,
    std::enable_if_t<std::is_integral<T>::value && !std::is_signed<T>::value &&
                     !std::is_same<T, uint64_t>::value &&
                     !std::is_same<T, bool>::value>* = nullptr>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           T value) {
  writer.Uint(value);
  return writer;
}

template <typename OutputStream, typename SourceEncoding,
          typename TargetEncoding, typename StackAllocator, typename T,
          std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           T value) {
  writer.Double(value);
  return writer;
}

template <typename OutputStream, typename SourceEncoding,
          typename TargetEncoding, typename StackAllocator>
Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
operator<<(Writer<OutputStream, SourceEncoding, TargetEncoding, StackAllocator>&
               writer,
           std::string_view value) {
  writer.String(value.data());
  return writer;
}

}  // namespace rapidjson

#endif  // ZKX_BASE_JSON_JSON_SERDE_H_
