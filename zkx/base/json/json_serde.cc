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
