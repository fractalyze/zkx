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

#include "zkx/base/json/json.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace zkx::base {
namespace {

struct SimpleData {
  std::string message;
  int index = 0;
  bool flag = false;
  std::vector<unsigned int> data;

  bool operator==(const SimpleData& other) const {
    return message == other.message && index == other.index &&
           flag == other.flag && data == other.data;
  }
  bool operator!=(const SimpleData& other) const { return !operator==(other); }
};

class JsonTest : public testing::Test {
 public:
  void SetUp() override {
    expected_simple_data_.message = "hello world";
    expected_simple_data_.index = 1;
    expected_simple_data_.flag = true;
    expected_simple_data_.data = std::vector<unsigned int>{0, 2, 4};
  }

 protected:
  SimpleData expected_simple_data_;
};

}  // namespace

template <>
class JsonSerde<SimpleData> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(const SimpleData& value, Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    // NOTE: avoid unnecessary copy.
    std::string_view message = value.message;
    AddJsonElement(object, "message", message, allocator);
    AddJsonElement(object, "index", value.index, allocator);
    AddJsonElement(object, "flag", value.flag, allocator);
    AddJsonElement(object, "data", value.data, allocator);
    return object;
  }

  static absl::StatusOr<SimpleData> To(const rapidjson::Value& json_value,
                                       std::string_view key) {
    TF_ASSIGN_OR_RETURN(std::string message,
                        ParseJsonElement<std::string>(json_value, "message"));
    TF_ASSIGN_OR_RETURN(int index, ParseJsonElement<int>(json_value, "index"));
    TF_ASSIGN_OR_RETURN(bool flag, ParseJsonElement<bool>(json_value, "flag"));
    TF_ASSIGN_OR_RETURN(
        std::vector<unsigned int> data,
        ParseJsonElement<std::vector<unsigned int>>(json_value, "data"));
    return SimpleData{std::move(message), index, flag, std::move(data)};
  }
};

TEST_F(JsonTest, LoadAndParseJson) {
  TF_ASSERT_OK_AND_ASSIGN(
      SimpleData simple_data,
      LoadAndParseJson<SimpleData>("zkx/base/json/test/simple_data.json"));
  EXPECT_EQ(simple_data, expected_simple_data_);
}

TEST_F(JsonTest, ParseInvalidJson) {
  // missing key
  std::string json = R"({})";
  absl::StatusOr<SimpleData> status_or = ParseJson<SimpleData>(json);
  EXPECT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().message(), "\"message\" key is not found");

  // invalid value
  json = R"({"message":3})";
  status_or = ParseJson<SimpleData>(json);
  EXPECT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().message(),
            "\"message\" expects type \"string\" but type \"number\" comes");
}

TEST_F(JsonTest, WriteToJson) {
  std::string json = WriteToJson(expected_simple_data_);
  EXPECT_EQ(
      json,
      R"({"message":"hello world","index":1,"flag":true,"data":[0,2,4]})");
}

}  // namespace zkx::base
