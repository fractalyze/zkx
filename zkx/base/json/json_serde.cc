#include "zkx/base/json/json_serde.h"

namespace zkx::base {

namespace {

constexpr const char* kRapidJsonTypeNames[] = {
    "null", "false", "true", "object", "array", "string", "number"};

}  // namespace

std::string RapidJsonMismatchedTypeError(std::string_view key,
                                         std::string_view type,
                                         const rapidjson::Value& value) {
  return absl::Substitute("\"$0\" expects type \"$1\" but type \"$2\" comes",
                          key, type, kRapidJsonTypeNames[value.GetType()]);
}

// static
bool JsonSerde<float>::s_allow_lossy_conversion = true;

// static
absl::StatusOr<float> JsonSerde<float>::To(const rapidjson::Value& json_value,
                                           std::string_view key) {
  if (!json_value.IsDouble()) {
    return absl::InvalidArgumentError(
        RapidJsonMismatchedTypeError(key, "float", json_value));
  }
  double value = json_value.GetDouble();
  if (s_allow_lossy_conversion) {
    if (!json_value.IsFloat()) {
      return absl::OutOfRangeError(RapidJsonOutOfRangeError(key, value));
    }

  } else {
    if (!json_value.IsLosslessFloat()) {
      return absl::OutOfRangeError(RapidJsonOutOfRangeError(key, value));
    }
  }
  return static_cast<float>(value);
}

// static
absl::StatusOr<double> JsonSerde<double>::To(const rapidjson::Value& json_value,
                                             std::string_view key) {
  if (!json_value.IsDouble()) {
    return absl::InvalidArgumentError(
        RapidJsonMismatchedTypeError(key, "double", json_value));
  }
  return json_value.GetDouble();
}

}  // namespace zkx::base
