// Copyright (c) 2020 The Console Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE.console file.

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

#include "zkx/base/flag/flag.h"

#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"

namespace zkx::base {

using ::absl_testing::StatusIs;

TEST(FlagTest, ShortName) {
  bool value;
  BoolFlag flag(&value);
  flag.set_short_name("-a");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("--a");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("a");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("-ab");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("-1");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("-_");
  EXPECT_EQ(flag.short_name(), "-a");
  flag.set_short_name("-b");
  EXPECT_EQ(flag.short_name(), "-b");
}

TEST(FlagTest, LongName) {
  bool value;
  BoolFlag flag(&value);
  flag.set_long_name("--a");
  EXPECT_EQ(flag.long_name(), "--a");
  flag.set_long_name("-a");
  EXPECT_EQ(flag.long_name(), "--a");
  flag.set_long_name("a");
  EXPECT_EQ(flag.long_name(), "--a");
  flag.set_long_name("--1");
  EXPECT_EQ(flag.long_name(), "--a");
  flag.set_long_name("--_");
  EXPECT_EQ(flag.long_name(), "--a");
  flag.set_long_name("--a_");
  EXPECT_EQ(flag.long_name(), "--a_");
}

TEST(FlagTest, Name) {
  bool value;
  BoolFlag flag(&value);
  flag.set_name("a");
  EXPECT_EQ(flag.name(), "a");
  flag.set_name("-a");
  EXPECT_EQ(flag.name(), "a");
  flag.set_name("--a");
  EXPECT_EQ(flag.name(), "a");
  flag.set_name("1");
  EXPECT_EQ(flag.name(), "a");
  flag.set_name("_");
  EXPECT_EQ(flag.name(), "a");
  flag.set_name("a_");
  EXPECT_EQ(flag.name(), "a_");
}

TEST(FlagTest, ParseValue) {
  bool bool_value = false;
  BoolFlag bool_flag(&bool_value);
  TF_EXPECT_OK(bool_flag.ParseValue(""));
  EXPECT_TRUE(bool_value);

  int16_t int16_value;
  Int16Flag int16_flag(&int16_value);
  TF_EXPECT_OK(int16_flag.ParseValue("123"));
  EXPECT_EQ(int16_value, 123);
  EXPECT_THAT(int16_flag.ParseValue("a"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "failed to convert int (\"a\")"));
  EXPECT_THAT(
      int16_flag.ParseValue("40000"),
      StatusIs(absl::StatusCode::kOutOfRange, "40000 is out of its range"));
  EXPECT_EQ(int16_value, 123);

  std::string string_value;
  StringFlag string_flag(&string_value);
  TF_EXPECT_OK(string_flag.ParseValue("abc"));
  EXPECT_EQ(string_value, "abc");
  EXPECT_THAT(string_flag.ParseValue(""),
              StatusIs(absl::StatusCode::kInvalidArgument, "input is empty"));
  EXPECT_EQ(string_value, "abc");

  std::string choice_value;
  StringChoicesFlag choices_flag(
      &choice_value, std::vector<std::string>{"cat", "dog", "duck"});
  TF_EXPECT_OK(choices_flag.ParseValue("cat"));
  TF_EXPECT_OK(choices_flag.ParseValue("dog"));
  TF_EXPECT_OK(choices_flag.ParseValue("duck"));
  EXPECT_THAT(choices_flag.ParseValue("bird"),
              StatusIs(absl::StatusCode::kNotFound, "bird is not in choices"));

  int32_t int32_value;
  Int32RangeFlag int32_range_flag(&int32_value, 1, 5);
  int32_range_flag.set_less_than_or_equal_to(true);
  TF_EXPECT_OK(int32_range_flag.ParseValue("2"));
  TF_EXPECT_OK(int32_range_flag.ParseValue("5"));
  EXPECT_THAT(int32_range_flag.ParseValue("1"),
              StatusIs(absl::StatusCode::kOutOfRange, "1 is out of range"));
  EXPECT_THAT(int32_range_flag.ParseValue("6"),
              StatusIs(absl::StatusCode::kOutOfRange, "6 is out of range"));
}

}  // namespace zkx::base
