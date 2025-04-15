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

#include "xla/tsl/platform/path.h"

#include <string>

#include "gtest/gtest.h"

#include "xla/tsl/platform/env.h"

namespace tsl::io {

// TODO(chokobole): Uncomment this. Dependency: JoinPath
// TEST(PathTest, JoinPath) {

// TODO(chokobole): Uncomment this. Dependency: IsAbsolutePath
// TEST(PathTest, IsAbsolutePath) {

// TODO(chokobole): Uncomment this. Dependency: Dirname
// TEST(PathTest, Dirname) {

// TODO(chokobole): Uncomment this. Dependency: Basename
// TEST(PathTest, Basename) {

// TODO(chokobole): Uncomment this. Dependency: Extension
// TEST(PathTest, Extension) {

// TODO(chokobole): Uncomment this. Dependency: CleanPath
// TEST(PathTest, CleanPath) {

// TODO(chokobole): Uncomment this. Dependency: CreateParseURI
// TEST(PathTest, CreateParseURI) {

// TODO(chokobole): Uncomment this. Dependency: CommonPathPrefix
// TEST(PathTest, CommonPathPrefix) {

// TODO(chokobole): Uncomment this. Dependency: GetTestWorkspaceDir
// TEST(PathTest, GetTestWorkspaceDir) {

TEST(PathTest, GetTestUndeclaredOutputsDir) {
  constexpr std::string_view kOriginalValue = "original value";
  std::string dir;

  dir = kOriginalValue;
  tsl::setenv("TEST_UNDECLARED_OUTPUTS_DIR", "/test/outputs",
              /*overwrite=*/true);
  EXPECT_TRUE(GetTestUndeclaredOutputsDir(&dir));
  EXPECT_EQ(dir, "/test/outputs");
  EXPECT_TRUE(GetTestUndeclaredOutputsDir(nullptr));

  dir = kOriginalValue;
  tsl::unsetenv("TEST_UNDECLARED_OUTPUTS_DIR");
  EXPECT_FALSE(GetTestUndeclaredOutputsDir(&dir));
  EXPECT_EQ(dir, kOriginalValue);
  EXPECT_FALSE(GetTestUndeclaredOutputsDir(nullptr));
}

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ResolveTestPrefixesKeepsThePathUnchanged
// clang-format on
// TEST(PathTest, ResolveTestPrefixesKeepsThePathUnchanged) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ResolveTestPrefixesCanResolveTestWorkspace
// clang-format on
// TEST(PathTest, ResolveTestPrefixesCanResolveTestWorkspace) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ResolveTestPrefixesCannotResolveTestWorkspace
// clang-format on
// TEST(PathTest, ResolveTestPrefixesCannotResolveTestWorkspace) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ResolveTestPrefixesCanResolveTestUndeclaredOutputsDir
// clang-format on
// TEST(PathTest, ResolveTestPrefixesCanResolveTestUndeclaredOutputsDir) {

// clang-format off
// TODO(chokobole): Uncomment this. Dependency: ResolveTestPrefixesCannotResolveTestUndeclaredOutputsDir
// clang-format on
// TEST(PathTest, ResolveTestPrefixesCannotResolveTestUndeclaredOutputsDir) {

}  // namespace tsl::io
