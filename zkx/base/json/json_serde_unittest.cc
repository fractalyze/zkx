#include "zkx/base/json/json_serde.h"

#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace zkx::base {

using ::absl_testing::StatusIs;

TEST(JsonSerdeTest, Bool) {
  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  EXPECT_EQ(JsonSerde<bool>::From(true, allocator).GetBool(), true);
  EXPECT_EQ(JsonSerde<bool>::From(false, allocator).GetBool(), false);

  rapidjson::Value true_value(true);
  rapidjson::Value false_value(false);
  rapidjson::Value invalid_value(123);

  bool value;
  TF_ASSERT_OK_AND_ASSIGN(value, JsonSerde<bool>::To(true_value, ""));
  EXPECT_TRUE(value);
  TF_ASSERT_OK_AND_ASSIGN(value, JsonSerde<bool>::To(false_value, ""));
  EXPECT_FALSE(value);
  EXPECT_THAT(JsonSerde<bool>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, Int64) {
  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  int64_t test_value = 1234567890123456789LL;
  EXPECT_EQ(JsonSerde<int64_t>::From(test_value, allocator).GetInt64(),
            test_value);

  rapidjson::Value valid_int64_value(test_value);
  rapidjson::Value valid_int_value(123);
  rapidjson::Value valid_losable_value(int64_t{1} << 54);
  rapidjson::Value invalid_value("not a number");

  int64_t value;
  TF_ASSERT_OK_AND_ASSIGN(value, JsonSerde<int64_t>::To(valid_int64_value, ""));
  EXPECT_EQ(value, test_value);
  TF_ASSERT_OK_AND_ASSIGN(value, JsonSerde<int64_t>::To(valid_int_value, ""));
  EXPECT_EQ(value, 123);
  EXPECT_THAT(JsonSerde<int64_t>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, Uint64) {
  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  uint64_t test_value = 12345678901234567890ULL;
  EXPECT_EQ(JsonSerde<uint64_t>::From(test_value, allocator).GetUint64(),
            test_value);

  rapidjson::Value valid_uint64_value(test_value);
  rapidjson::Value valid_uint_value(123);
  rapidjson::Value invalid_value("not a number");

  uint64_t value;
  TF_ASSERT_OK_AND_ASSIGN(value,
                          JsonSerde<uint64_t>::To(valid_uint64_value, ""));
  EXPECT_EQ(value, test_value);
  TF_ASSERT_OK_AND_ASSIGN(value, JsonSerde<uint64_t>::To(valid_uint_value, ""));
  EXPECT_EQ(value, 123);
  EXPECT_THAT(JsonSerde<uint64_t>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, Int) {
  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  int test_value = 123456;
  EXPECT_EQ(JsonSerde<int>::From(test_value, allocator).GetInt(), test_value);

  rapidjson::Value valid_int_value(test_value);
  rapidjson::Value valid_int64_value(int64_t{1} << 31);
  rapidjson::Value invalid_value("not a number");

  int value;
  TF_ASSERT_OK_AND_ASSIGN(value, JsonSerde<int>::To(valid_int_value, ""));
  EXPECT_EQ(value, test_value);
  EXPECT_THAT(JsonSerde<int>::To(valid_int64_value, ""),
              StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(JsonSerde<int>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, Uint) {
  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  unsigned int test_value = 123456;
  EXPECT_EQ(JsonSerde<unsigned int>::From(test_value, allocator).GetUint(),
            test_value);

  rapidjson::Value valid_uint_value(test_value);
  rapidjson::Value valid_uint64_value(uint64_t{1} << 32);
  rapidjson::Value invalid_value("not a number");

  unsigned int value;
  TF_ASSERT_OK_AND_ASSIGN(value,
                          JsonSerde<unsigned int>::To(valid_uint_value, ""));
  EXPECT_EQ(value, test_value);
  EXPECT_THAT(JsonSerde<unsigned int>::To(valid_uint64_value, ""),
              StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(JsonSerde<unsigned int>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, Float) {
  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  float test_value = 123.456f;
  EXPECT_EQ(JsonSerde<float>::From(test_value, allocator).GetFloat(),
            test_value);

  rapidjson::Value valid_float_value(test_value);
  rapidjson::Value valid_double_value(3.4028234e38);
  rapidjson::Value invalid_value("not a number");

  float value;
  TF_ASSERT_OK_AND_ASSIGN(value, JsonSerde<float>::To(valid_float_value, ""));
  EXPECT_EQ(value, test_value);

  // Test with lossy conversion disabled
  JsonSerde<float>::s_allow_lossy_conversion = false;
  EXPECT_THAT(JsonSerde<float>::To(valid_double_value, ""),
              StatusIs(absl::StatusCode::kOutOfRange));

  // Test with lossy conversion enabled
  JsonSerde<float>::s_allow_lossy_conversion = true;
  TF_ASSERT_OK_AND_ASSIGN(value, JsonSerde<float>::To(valid_double_value, ""));
  EXPECT_FLOAT_EQ(value, 3.4028234e38);

  EXPECT_THAT(JsonSerde<float>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, Double) {
  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  double test_value = 123.456;
  EXPECT_EQ(JsonSerde<double>::From(test_value, allocator).GetDouble(),
            test_value);

  rapidjson::Value valid_double_value(test_value);
  rapidjson::Value invalid_value("not a number");

  double value;
  TF_ASSERT_OK_AND_ASSIGN(value, JsonSerde<double>::To(valid_double_value, ""));
  EXPECT_EQ(value, test_value);
  EXPECT_THAT(JsonSerde<double>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, String) {
  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  std::string test_value = "hello world";
  EXPECT_EQ(JsonSerde<std::string>::From(test_value, allocator).GetString(),
            test_value);

  rapidjson::Value valid_string_value(test_value, allocator);
  rapidjson::Value invalid_value(123);

  std::string value;
  TF_ASSERT_OK_AND_ASSIGN(value,
                          JsonSerde<std::string>::To(valid_string_value, ""));
  EXPECT_EQ(value, test_value);
  EXPECT_THAT(JsonSerde<std::string>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, StringView) {
  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  std::string_view test_value = "hello world";
  EXPECT_EQ(
      JsonSerde<std::string_view>::From(test_value, allocator).GetString(),
      test_value);

  rapidjson::Value valid_string_view_value(test_value.data(),
                                           test_value.length(), allocator);
  rapidjson::Value invalid_value(123);

  std::string_view value;
  TF_ASSERT_OK_AND_ASSIGN(
      value, JsonSerde<std::string_view>::To(valid_string_view_value, ""));
  EXPECT_EQ(value, test_value);
  EXPECT_THAT(JsonSerde<std::string_view>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, Enum) {
  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  enum class TestEnum { kValue1 = 1, kValue2 = 2 };

  TestEnum test_value = TestEnum::kValue1;
  EXPECT_EQ(JsonSerde<TestEnum>::From(test_value, allocator).GetInt(),
            static_cast<int>(test_value));

  rapidjson::Value valid_enum_value(static_cast<int>(test_value));
  rapidjson::Value invalid_value("not an enum");

  TestEnum value;
  TF_ASSERT_OK_AND_ASSIGN(value, JsonSerde<TestEnum>::To(valid_enum_value, ""));
  EXPECT_EQ(value, test_value);
  EXPECT_THAT(JsonSerde<TestEnum>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, Array) {
  using ArrayType = std::array<int, 3>;

  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  ArrayType test_value = {1, 2, 3};
  rapidjson::Value array_value =
      JsonSerde<ArrayType>::From(test_value, allocator);
  EXPECT_TRUE(array_value.IsArray());
  EXPECT_EQ(array_value.Size(), 3);
  EXPECT_EQ(array_value[0].GetInt(), 1);
  EXPECT_EQ(array_value[1].GetInt(), 2);
  EXPECT_EQ(array_value[2].GetInt(), 3);

  rapidjson::Value valid_array_value(rapidjson::kArrayType);
  valid_array_value.PushBack(1, allocator);
  valid_array_value.PushBack(2, allocator);
  valid_array_value.PushBack(3, allocator);

  rapidjson::Value invalid_value(123);

  ArrayType value;
  TF_ASSERT_OK_AND_ASSIGN(value,
                          JsonSerde<ArrayType>::To(valid_array_value, ""));
  EXPECT_EQ(value, test_value);

  valid_array_value.PushBack(4, allocator);
  EXPECT_THAT(JsonSerde<ArrayType>::To(valid_array_value, ""),
              StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(JsonSerde<ArrayType>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, Vector) {
  using VectorType = std::vector<int>;

  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  VectorType test_value = {1, 2, 3};
  rapidjson::Value array_value =
      JsonSerde<VectorType>::From(test_value, allocator);
  EXPECT_TRUE(array_value.IsArray());
  EXPECT_EQ(array_value.Size(), 3);
  EXPECT_EQ(array_value[0].GetInt(), 1);
  EXPECT_EQ(array_value[1].GetInt(), 2);
  EXPECT_EQ(array_value[2].GetInt(), 3);

  rapidjson::Value valid_array_value(rapidjson::kArrayType);
  valid_array_value.PushBack(1, allocator);
  valid_array_value.PushBack(2, allocator);
  valid_array_value.PushBack(3, allocator);

  rapidjson::Value invalid_value(123);

  VectorType value;
  TF_ASSERT_OK_AND_ASSIGN(value,
                          JsonSerde<VectorType>::To(valid_array_value, ""));
  EXPECT_EQ(value, test_value);

  EXPECT_THAT(JsonSerde<VectorType>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(JsonSerdeTest, Optional) {
  using OptionalType = std::optional<int>;

  rapidjson::Document document;
  auto& allocator = document.GetAllocator();

  // Test with value
  OptionalType test_value = 42;
  rapidjson::Value value = JsonSerde<OptionalType>::From(test_value, allocator);
  EXPECT_TRUE(value.IsInt());
  EXPECT_EQ(value.GetInt(), 42);

  // Test with null
  OptionalType null_value = std::nullopt;
  rapidjson::Value null_json =
      JsonSerde<OptionalType>::From(null_value, allocator);
  EXPECT_TRUE(null_json.IsNull());

  // Test conversion back with value
  rapidjson::Value valid_value(42);
  OptionalType result;
  TF_ASSERT_OK_AND_ASSIGN(result, JsonSerde<OptionalType>::To(valid_value, ""));
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 42);

  // Test conversion back with null
  rapidjson::Value null_value_json;
  TF_ASSERT_OK_AND_ASSIGN(result,
                          JsonSerde<OptionalType>::To(null_value_json, ""));
  EXPECT_FALSE(result.has_value());

  // Test invalid type
  rapidjson::Value invalid_value("not an int");
  EXPECT_THAT(JsonSerde<OptionalType>::To(invalid_value, ""),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace zkx::base
