// Copyright (c) 2020 The Console Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "zkx/base/flag/flag_parser.h"

#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/env.h"

namespace zkx::base {

using ::absl_testing::StatusIs;

#define PARSE(...)                    \
  const char* argv[] = {__VA_ARGS__}; \
  std::string error;                  \
  absl::Status s = parser.Parse(std::size(argv), const_cast<char**>(argv))

#define PARSE_KNOWN(...)              \
  const char* argv[] = {__VA_ARGS__}; \
  int argc = std::size(argv);         \
  std::string error;                  \
  absl::Status s = parser.ParseKnown(&argc, const_cast<char**>(argv))

#define PARSE_WITH_FORWARD(...)       \
  const char* argv[] = {__VA_ARGS__}; \
  int argc = std::size(argv);         \
  std::string error;                  \
  std::vector<std::string> forward;   \
  absl::Status s =                    \
      parser.ParseWithForward(argc, const_cast<char**>(argv), &forward)

TEST(FlagParserTest, ValidateInternally) {
  {
    FlagParser parser;
    uint16_t value;
    parser.AddFlag<Uint16Flag>(&value);
    EXPECT_THAT(parser.ValidateInternally(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         "Flag should be positional or optional."));
  }
  {
    FlagParser parser;
    uint16_t value;
    parser.AddFlag<Uint16Flag>(&value).set_name("value").set_short_name("-v");
    EXPECT_THAT(parser.ValidateInternally(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         "\"value\" is both positional and optional, please "
                         "choose either one of them."));
  }
  {
    FlagParser parser;
    parser.AddSubParser().set_short_name("-a");
    EXPECT_THAT(parser.ValidateInternally(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         "Subparser \"-a\" should be positional."));
  }
  {
    FlagParser parser;
    uint16_t value;
    parser.AddFlag<Uint16Flag>(&value).set_short_name("-v");
    parser.AddFlag<Uint16Flag>(&value).set_name("value");
    EXPECT_THAT(parser.ValidateInternally(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         "\"value\" should be before any optional arguments."));
  }
  {
    FlagParser parser;
    bool value;
    parser.AddFlag<BoolFlag>(&value).set_name("value");
    EXPECT_THAT(
        parser.ValidateInternally(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 "\"value\" can't parse a value, how about considering using "
                 "set_short_name() or set_long_name()?"));
  }
  {
    FlagParser parser;
    uint16_t value;
    parser.AddSubParser().set_name("a");
    parser.AddFlag<Uint16Flag>(&value).set_name("value");
    EXPECT_THAT(parser.ValidateInternally(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         "\"value\" can't be positional if the parser has "
                         "subparser, how about considering using "
                         "set_short_name() or set_long_name()?"));
  }
  {
    FlagParser parser;
    uint16_t value;
    parser.AddFlag<Uint16Flag>(&value).set_short_name("-a");
    SubParser& sub_parser = parser.AddSubParser();
    sub_parser.set_name("test");
    sub_parser.AddFlag<Uint16Flag>(&value).set_name("name");
    EXPECT_THAT(parser.ValidateInternally(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         "SubParser should be at the very front."));
  }
}

TEST(FlagParserTest, UndefinedArgument) {
  FlagParser parser;
  uint16_t value;
  parser.AddFlag<Uint16Flag>(&value).set_long_name("--value");
  {
    PARSE("program", "--v", "16");
    EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument,
                            "Met unknown argument: \"--v\"."));
  }
  {
    PARSE("program", "--val", "16");
    EXPECT_THAT(
        s, StatusIs(
               absl::StatusCode::kInvalidArgument,
               "Met unknown argument: \"--val\", maybe you mean \"--value\"?"));
  }
}

TEST(FlagParserTest, DefaultValue) {
  FlagParser parser;
  uint16_t value;
  parser.AddFlag<Uint16Flag>(&value)
      .set_default_value(uint16_t{12})
      .set_short_name("-v");
  {
    PARSE("program");
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, 12);
  }
}

TEST(FlagParserTest, PositionalArguments) {
  FlagParser parser;
  uint16_t value = 0;
  uint16_t value2 = 0;
  parser.AddFlag<Uint16Flag>(&value).set_name("flag");
  parser.AddFlag<Uint16Flag>(&value2).set_name("flag2");
  {
    PARSE("program", "12");
    EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument,
                            "\"flag2\" is positional, but not set."));
  }
  {
    PARSE("program", "12", "34");
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, 12);
    EXPECT_EQ(value2, 34);
  }
}

TEST(FlagParserTest, RequiredOptionalArguments) {
  FlagParser parser;
  uint16_t value = 0;
  uint16_t value2 = 0;
  parser.AddFlag<Uint16Flag>(&value).set_short_name("-a");
  parser.AddFlag<Uint16Flag>(&value2).set_short_name("-b").set_required();
  {
    PARSE("program", "-a", "12");
    EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument,
                            "\"-b\" is required, but not set."));
    EXPECT_EQ(value, 12);
  }
  {
    value = 0;
    PARSE("program", "-b", "34");
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, 0);
    EXPECT_EQ(value2, 34);
  }
  {
    PARSE("program", "-a", "56", "-b", "78");
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, 56);
    EXPECT_EQ(value2, 78);
  }
}

TEST(FlagParserTest, ConcatenatedOptionalFlags) {
  FlagParser parser;
  bool value = false;
  bool value2 = false;
  int32_t value3 = 0;
  parser.AddFlag<BoolFlag>(&value).set_short_name("-a");
  parser.AddFlag<BoolFlag>(&value2).set_short_name("-b");
  parser.AddFlag<Int32Flag>(&value3).set_short_name("-c");
  {
    PARSE("program", "-ab");
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(value);
    EXPECT_TRUE(value2);
  }
  {
    PARSE("program", "-ac");
    EXPECT_THAT(
        s, StatusIs(absl::StatusCode::kInvalidArgument,
                    "Met unknown argument: \"-ac\", maybe you mean \"-a\"?"));
  }
}

TEST(FlagParserTest, VectorFlag) {
  FlagParser parser;
  std::vector<int> numbers;
  parser.AddFlag<Flag<std::vector<int>>>(&numbers).set_short_name("-a");
  {
    PARSE("program", "-a", "1", "-a", "2", "-a", "3");
    EXPECT_TRUE(s.ok());
    EXPECT_THAT(numbers, testing::ElementsAre(1, 2, 3));
  }
}

TEST(FlagParserTest, CustomParseValueCallback) {
  FlagParser parser;
  std::string value;
  parser
      .AddFlag<StringFlag>(&value,
                           [](std::string_view arg, std::string* value) {
                             if (arg == "cat" || arg == "dog") {
                               *value = std::string(arg);
                               return absl::OkStatus();
                             }
                             return absl::InvalidArgumentError(absl::Substitute(
                                 "$0 is not either cat or dog", arg));
                           })
      .set_long_name("--animal");
  {
    PARSE("program", "--animal", "pig");
    EXPECT_THAT(s, StatusIs(absl::StatusCode::kInvalidArgument,
                            "Failed to parse \"--animal\": (reason: pig is not "
                            "either cat or dog)."));
  }
  {
    PARSE("program", "--animal", "cat");
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, "cat");
  }
}

TEST(FlagParserTest, ParseValueFromEnvironment) {
  FlagParser parser;
  std::string value;
  parser.AddFlag<StringFlag>(&value).set_env_name("VALUE").set_short_name("-v");
  {
    tsl::setenv("VALUE", "value", /*overwrite=*/1);
    PARSE("program");
    EXPECT_EQ(value, "value");
  }
}

TEST(FlagParserTest, ParseKnown) {
  FlagParser parser;
  int value = 0;
  parser.AddFlag<IntFlag>(&value).set_short_name("-a");
  bool value2 = false;
  parser.AddFlag<BoolFlag>(&value2).set_short_name("-b");
  {
    PARSE_KNOWN("program", "-a", "1", "--unknown", "-b");
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(value, 1);
    EXPECT_EQ(value2, true);
    EXPECT_EQ(argc, 2);
    EXPECT_STREQ(argv[0], "program");
    EXPECT_STREQ(argv[1], "--unknown");
  }

  {
    value = 0;
    value2 = false;
    PARSE_KNOWN("program", "-a", "1", "-b", "--unknown", "2");
    EXPECT_EQ(value, 1);
    EXPECT_EQ(value2, true);
    EXPECT_EQ(argc, 3);
    EXPECT_STREQ(argv[0], "program");
    EXPECT_STREQ(argv[1], "--unknown");
    EXPECT_STREQ(argv[2], "2");
  }
}

TEST(FlagParserTest, ParseWithForward) {
  FlagParser parser;
  int value = 0;
  parser.AddFlag<IntFlag>(&value).set_short_name("-a");
  bool value2 = false;
  parser.AddFlag<BoolFlag>(&value2).set_short_name("-b");
  {
    PARSE_WITH_FORWARD("program", "-a", "1", "--", "-b");
    EXPECT_EQ(value, 1);
    EXPECT_EQ(value2, false);
    EXPECT_THAT(forward, testing::ElementsAre("-b"));
  }
}

TEST(FlagParserTest, SubParserTest) {
  FlagParser parser;
  int a = 0;
  int b = 0;
  bool verbose = false;
  SubParser& add_parser = parser.AddSubParser().set_name("add");
  add_parser.AddFlag<Int32Flag>(&a).set_name("a");
  add_parser.AddFlag<Int32Flag>(&b).set_name("b");
  SubParser& sub_parser = parser.AddSubParser().set_name("sub");
  sub_parser.AddFlag<Int32Flag>(&a).set_name("a");
  sub_parser.AddFlag<Int32Flag>(&b).set_name("b");
  parser.AddFlag<BoolFlag>(&verbose).set_short_name("-v");
  {
    PARSE("program", "add", "1", "2");
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(add_parser.is_set());
    EXPECT_FALSE(sub_parser.is_set());
    EXPECT_EQ(1, a);
    EXPECT_EQ(2, b);
    EXPECT_FALSE(verbose);
  }

  add_parser.reset();
  {
    PARSE("program", "-v", "add", "1", "2");
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(add_parser.is_set());
    EXPECT_FALSE(sub_parser.is_set());
    EXPECT_EQ(1, a);
    EXPECT_EQ(2, b);
    EXPECT_TRUE(verbose);
  }
}

#undef PARSE
#undef PARSE_KNOWN
#undef PARSE_WITH_FORWARD

}  // namespace zkx::base
