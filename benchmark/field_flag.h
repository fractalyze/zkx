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

#ifndef BENCHMARK_FIELD_FLAG_H_
#define BENCHMARK_FIELD_FLAG_H_

#include "absl/status/status.h"

#include "zkx/base/flag/flag_parser.h"
#include "zkx/base/flag/flag_value_traits.h"

namespace zkx {
namespace benchmark {

enum class Field {
  kBn254Fr,
};

void AddFieldFlag(base::FlagParserBase& parser, benchmark::Field* field);

std::string_view FieldToHloString(Field field);

}  // namespace benchmark

namespace base {

template <>
class FlagValueTraits<benchmark::Field> {
 public:
  static absl::Status ParseValue(std::string_view input,
                                 benchmark::Field* value);
};

}  // namespace base
}  // namespace zkx

#endif  // BENCHMARK_FIELD_FLAG_H_
