/* Copyright 2017 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include <optional>
#include <random>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/text_format.h"

#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_module.h"

namespace zkx {

// Generates a vector of arguments containing fake data. The number, shape and
// layout of the arguments is appropriate for given HLO module.
//
// A best-effort attempt is made to generate the data in a way which produce
// stable computation results across platforms. Specifically:
//
//  (1) Init values of reductions should be the identity of the reduction
//  computation.
//
//  (2) Indices of dynamic slices and update slices should be in bounds.
//
//  (3) Keys of key/value sorts should contain no duplicates.
//
// These constraints are best-effort only.
//
// If max_bits_of_precision is set to a number, then floating point & integer
// types will be constrained to be represented in that number of bits. Setting
// it to 5 for integers would mean it only creates integers between -32 and 32.
//
// If pseudo_random is true, the generated numbers will be generated
// deterministically in a pseudo random way unless the values are constrated to
// be e.g. init values as above. If pseudo_random is false, the returned values
// will be generated in a faster way that yields less interesting data, e.g. the
// values may all be just the same value.
//
// TODO(b/79942829): Make interesting argument generation fast enough that using
// pseudo_random does not save any noticeable amount of time so that the
// parameter can be removed.
absl::StatusOr<std::vector<Literal>> MakeFakeArguments(
    const HloModule* module, bool pseudo_random = true,
    bool treat_gte_as_data_formatting = false,
    std::optional<int64_t> max_bits_of_precision = std::nullopt,
    std::minstd_rand0* engine = nullptr);

// Overload which accepts a random number generator. This enables generation of
// different random values with sequential calls to MakeFakeArguments by reusing
// the same generator.
absl::StatusOr<std::vector<Literal>> MakeFakeArguments(
    const HloModule* module, std::minstd_rand0* engine,
    bool treat_gte_as_data_formatting = false,
    std::optional<int64_t> max_bits_of_precision = std::nullopt);

// Creates a dot op with operands 'lhs' and 'rhs' that contracts dimension 1 of
// the LHS with dimension 0 of the RHS with no batch dimensions.
// Both LHS and the RHS must have rank less than or equal to 2.
std::unique_ptr<HloDotInstruction> CreateCanonicalDot(const Shape& shape,
                                                      HloInstruction* lhs,
                                                      HloInstruction* rhs);

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
