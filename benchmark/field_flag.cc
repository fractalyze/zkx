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

#include "benchmark/field_flag.h"

#include "absl/base/optimization.h"
#include "absl/strings/substitute.h"

#include "zkx/base/flag/flag.h"

namespace zkx {
namespace benchmark {

void AddFieldFlag(base::FlagParserBase& parser, benchmark::Field* field) {
  parser.AddFlag<base::Flag<benchmark::Field>>(field)
      .set_name("field")
      .set_help("Field to generate scalars for");
}

std::string_view FieldToHloString(Field field) {
  switch (field) {
    case Field::kBn254Fr:
      return "bn254.sf";
  }
  ABSL_UNREACHABLE();
  return "";
}

}  // namespace benchmark

namespace base {

// static
absl::Status FlagValueTraits<benchmark::Field>::ParseValue(
    std::string_view input, benchmark::Field* value) {
  if (input == "bn254-fr") {
    *value = benchmark::Field::kBn254Fr;
  } else {
    return absl::NotFoundError(absl::Substitute("Unknown field: $0", input));
  }
  return absl::OkStatus();
}

}  // namespace base
}  // namespace zkx
