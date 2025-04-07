/* Copyright 2017 The OpenXLA Authors.

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

#ifndef ZKX_TESTS_TEST_UTILS_H_
#define ZKX_TESTS_TEST_UTILS_H_

#include <stdint.h>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/text_format.h"

namespace zkx {

template <typename MessageType>
absl::StatusOr<MessageType> ParseTextProto(const std::string& text_proto) {
  google::protobuf::TextFormat::Parser parser;
  MessageType parsed_proto;
  google::protobuf::io::ArrayInputStream input_stream(
      text_proto.data(), static_cast<int32_t>(text_proto.size()));
  if (!parser.Parse(&input_stream, &parsed_proto)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not parse text proto: ", text_proto));
  }
  return parsed_proto;
}

}  // namespace zkx

#endif  // ZKX_TESTS_TEST_UTILS_H_
