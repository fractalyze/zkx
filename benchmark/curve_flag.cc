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

#include "benchmark/curve_flag.h"

#include "absl/base/optimization.h"
#include "absl/strings/substitute.h"

#include "zkx/base/flag/flag.h"

namespace zkx {
namespace benchmark {

void AddCurveFlag(base::FlagParserBase& parser, benchmark::Curve* curve) {
  parser.AddFlag<base::Flag<benchmark::Curve>>(curve)
      .set_name("curve")
      .set_help("Curve to use for the benchmark");
}

std::string_view CurveToScalarFieldHloString(Curve curve) {
  switch (curve) {
    case Curve::kBn254G1:
      return "bn254_sf";
    case Curve::kBn254G2:
      return "bn254_sf";
  }
  ABSL_UNREACHABLE();
  return "";
}

std::string_view CurveToHloString(Curve curve) {
  switch (curve) {
    case Curve::kBn254G1:
      return "bn254_g1";
    case Curve::kBn254G2:
      return "bn254_g2";
  }
  ABSL_UNREACHABLE();
  return "";
}

}  // namespace benchmark

namespace base {

// static
absl::Status FlagValueTraits<benchmark::Curve>::ParseValue(
    std::string_view input, benchmark::Curve* value) {
  if (input == "bn254-g1") {
    *value = benchmark::Curve::kBn254G1;
  } else if (input == "bn254-g2") {
    *value = benchmark::Curve::kBn254G2;
  } else {
    return absl::NotFoundError(absl::Substitute("Unknown curve: $0", input));
  }
  return absl::OkStatus();
}

}  // namespace base
}  // namespace zkx
