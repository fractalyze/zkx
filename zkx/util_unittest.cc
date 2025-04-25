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

#include "zkx/util.h"

#include <vector>

#include "gtest/gtest.h"

namespace zkx {

// TODO(chokobole): Add test. Dependency: Reindent
// TEST(UtilTest, ReindentsDifferentNumberOfLeadingSpacesUniformly) {

// TODO(chokobole): Add test. Dependency: HumanReadableNumFlops
// TEST(UtilTest, HumanReadableNumFlopsExample) {

// TODO(chokobole): Add test. Dependency: CommaSeparatedString
// TEST(UtilTest, CommaSeparatedString) {

// TODO(chokobole): Add test. Dependency: VectorString
// TEST(UtilTest, VectorString) {

// TODO(chokobole): Add test. Dependency: LogLines
// TEST(UtilTest, LogLines) {

TEST(UtilTest, CommonFactors) {
  struct {
    std::vector<int64_t> a, b;
    absl::InlinedVector<std::pair<int64_t, int64_t>, 8> expected;
  } test_cases[] = {
      {/*.a =*/{0}, /*.b =*/{0}, /*.expected =*/{{0, 0}, {1, 1}}},
      {/*.a =*/{1}, /*.b =*/{}, /*.expected =*/{{0, 0}, {1, 0}}},
      {/*.a =*/{}, /*.b =*/{1}, /*.expected =*/{{0, 0}, {0, 1}}},
      {/*.a =*/{0, 10}, /*.b =*/{0, 10, 3}, /*.expected =*/{{0, 0}, {2, 3}}},
      {/*.a =*/{1, 0}, /*.b =*/{1, 0, 1},
       /*.expected =*/{{0, 0}, {1, 1}, {2, 2}, {2, 3}}},
      {/*.a =*/{0, 1}, /*.b =*/{0, 1}, /*.expected =*/{{0, 0}, {1, 1}, {2, 2}}},
      {/*.a =*/{}, /*.b =*/{}, /*.expected =*/{{0, 0}}},
      {/*.a =*/{2, 5, 1, 3},
       /*.b =*/{1, 10, 3, 1},
       /*.expected =*/{{0, 0}, {0, 1}, {2, 2}, {3, 2}, {4, 3}, {4, 4}}},
      {/*.a =*/{1, 1, 3},
       /*.b =*/{1, 1, 3},
       /*.expected =*/{{0, 0}, {1, 1}, {2, 2}, {3, 3}}},
      // Splitting and combining dimensions.
      {/*.a =*/{2, 6},
       /*.b =*/{4, 3},
       /*.expected =*/{{0, 0}, {2, 2}}},
      {/*.a =*/{1, 2, 6},
       /*.b =*/{4, 1, 3, 1},
       /*.expected =*/{{0, 0}, {1, 0}, {3, 3}, {3, 4}}},
      // Extra degenerated dimension (second and third dims in the output) forms
      // single common factor group.
      {/*.a =*/{1, 2, 1},
       /*.b =*/{1, 1, 1, 2},
       /*.expected =*/{{0, 0}, {1, 1}, {1, 2}, {1, 3}, {2, 4}, {3, 4}}}};
  for (const auto& test_case : test_cases) {
    EXPECT_EQ(test_case.expected, CommonFactors(test_case.a, test_case.b));
  }
}

TEST(UtilTest, SanitizeFileName) {
  EXPECT_EQ(SanitizeFileName(""), "");
  EXPECT_EQ(SanitizeFileName("abc"), "abc");
  EXPECT_EQ(SanitizeFileName("/\\[]"), "____");
  EXPECT_EQ(SanitizeFileName("/A\\B[C]"), "_A_B_C_");
}

// TODO(chokobole): Add test. Dependency: RoundTripFpToString
// TEST(UtilTest, RoundTripFpToString) {

// TODO(chokobole): Add test. Dependency: UnpackIntN
// TEST(UtilTest, PackInt4) {

}  // namespace zkx
