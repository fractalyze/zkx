/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/platform/errors.h"

namespace tsl::errors::internal {

std::unordered_map<std::string, std::string> GetPayloads(
    const absl::Status& status) {
  std::unordered_map<std::string, std::string> payloads;
  status.ForEachPayload(
      [&payloads](std::string_view key, const absl::Cord& value) {
        payloads[std::string(key)] = std::string(value);
      });
  return payloads;
}

// Inserts all given payloads into the given status. Will overwrite existing
// payloads if they exist with the same key.
void InsertPayloads(
    absl::Status& status,
    const std::unordered_map<std::string, std::string>& payloads) {
  for (const auto& payload : payloads) {
    status.SetPayload(payload.first, absl::Cord(payload.second));
  }
}

// Copies all payloads from one Status to another. Will overwrite existing
// payloads in the destination if they exist with the same key.
void CopyPayloads(const absl::Status& from, absl::Status& to) {
  from.ForEachPayload([&to](std::string_view key, const absl::Cord& value) {
    to.SetPayload(key, value);
  });
}

absl::Status Create(
    absl::StatusCode code, std::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  absl::Status status(code, message);
  InsertPayloads(status, payloads);
  return status;
}

// Returns a new Status, replacing its message with the given.
absl::Status CreateWithUpdatedMessage(const absl::Status& status,
                                      std::string_view message) {
  return Create(status.code(), message, GetPayloads(status));
}

}  // namespace tsl::errors::internal
