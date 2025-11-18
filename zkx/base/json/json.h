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

#ifndef ZKX_BASE_JSON_JSON_H_
#define ZKX_BASE_JSON_JSON_H_

#include <string>

#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "zkx/base/json/json_serde.h"

namespace zkx::base {

// Parse json string and return json value on success.
template <typename T>
absl::StatusOr<T> ParseJson(std::string_view content) {
  rapidjson::Document document;
  document.Parse(content.data(), content.length());
  if (document.HasParseError()) {
    return absl::InvalidArgumentError(
        absl::Substitute("Failed to parse with error \"$0\" at offset $1",
                         rapidjson::GetParseError_En(document.GetParseError()),
                         document.GetErrorOffset()));
  }
  return JsonSerde<T>::To(document.GetObject(), "");
}

// Load from file and parse json string and return json value on success.
template <typename T>
absl::StatusOr<T> LoadAndParseJson(const std::string& path) {
  std::string content;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), path, &content));
  return ParseJson<T>(content);
}

// Write json value to json string.
template <typename T>
std::string WriteToJson(const T& value) {
  rapidjson::Document document;
  rapidjson::Value json_value =
      JsonSerde<T>::From(value, document.GetAllocator());
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  json_value.Accept(writer);
  return buffer.GetString();
}

}  // namespace zkx::base

#endif  // ZKX_BASE_JSON_JSON_H_
