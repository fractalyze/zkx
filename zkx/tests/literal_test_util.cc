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

#include "zkx/tests/literal_test_util.h"

#include "zkx/literal_comparison.h"

namespace zkx {

namespace {

::testing::AssertionResult StatusToAssertion(const absl::Status& s) {
  if (s.ok()) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << s.message();
}

}  // namespace

// static
::testing::AssertionResult LiteralTestUtil::Equal(const LiteralSlice& expected,
                                                  const LiteralSlice& actual) {
  return StatusToAssertion(literal_comparison::Equal(expected, actual));
}

}  // namespace zkx
