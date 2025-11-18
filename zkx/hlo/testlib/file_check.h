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

#ifndef ZKX_HLO_TESTLIB_FILE_CHECK_H_
#define ZKX_HLO_TESTLIB_FILE_CHECK_H_

#include <string>

#include "absl/status/statusor.h"

namespace zkx {

// Runs FileCheck with the given pattern over given input string. Provided that
// FileCheck can execute, returns true if and only if FileCheck succeeded in
// matching the input.
absl::StatusOr<bool> RunFileCheck(const std::string& input,
                                  std::string_view pattern);

// Runs FileCheck with the given pattern file over given input string. Provided
// that FileCheck can execute, returns true if and only if FileCheck succeeded
// in matching the input.
absl::StatusOr<bool> RunFileCheckWithPatternFile(
    const std::string& input, const std::string& pattern_file);

}  // namespace zkx

#endif  // ZKX_HLO_TESTLIB_FILE_CHECK_H_
