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

#include "zkx/primitive_util.h"

#include <stddef.h>

#include "absl/log/check.h"
#include "absl/strings/ascii.h"

namespace zkx::primitive_util {
namespace {

// Class to memoize the computation of
//   absl::AsciiStrToLower(PrimitiveType_Name(p))
// for all PrimitiveType values "p"
//
// zkx::OPAQUE_TYPE canonically maps to the string "opaque" -- the only reason
// it's called OPAQUE_TYPE is to avoid clashing with a windows.h macro.
class PrimitiveTypeNameGenerator {
 public:
  PrimitiveTypeNameGenerator() {
    for (size_t idx = 0; idx < std::size(lowercase_name_); ++idx) {
      PrimitiveType t = static_cast<PrimitiveType>(idx + PrimitiveType_MIN);
      if (t == OPAQUE_TYPE) {
        lowercase_name_[idx] = "opaque";
      } else if (PrimitiveType_IsValid(t)) {
        lowercase_name_[idx] = absl::AsciiStrToLower(PrimitiveType_Name(t));
      }
    }
  }
  std::string_view LowercaseName(PrimitiveType t) {
    CHECK_GE(t, PrimitiveType_MIN);
    CHECK_LE(t, PrimitiveType_MAX);
    CHECK(PrimitiveType_IsValid(t))
        << "Invalid PrimitiveType: " << static_cast<int>(t);
    return lowercase_name_[t - PrimitiveType_MIN];
  }

 private:
  std::string lowercase_name_[PrimitiveType_MAX - PrimitiveType_MIN + 1];
};

}  // namespace

std::string_view LowercasePrimitiveTypeName(PrimitiveType s) {
  static auto* gen = new PrimitiveTypeNameGenerator();
  return gen->LowercaseName(s);
}

}  // namespace zkx::primitive_util
