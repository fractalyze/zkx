#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "zkx/base/buffer/serde.h"
#include "zkx/base/buffer/vector_buffer.h"

namespace zkx::base {

enum class Color {
  kRed,
  kBlue,
  kGreen,
};

TEST(BufferTest, Write) {
  constexpr char kCharValue = 'c';
  constexpr int kIntValue = 12345;
  constexpr bool kBooleanValue = true;
  Color kColor = Color::kBlue;
  const char* kCharPtrValue = "abc";
  std::string kStringValue = "def";
  uint64_t kIntBoundedArray[4] = {1, 2, 3, 4};
  std::vector<int> kIntVector = {5, 6, 7};
  std::array<int, 4> kIntArray = {8, 9, 10, 11};
  std::tuple<int, int, int> kTuple = std::make_tuple(1, 2, 3);

  for (Endian endian : {Endian::kNative, Endian::kBig, Endian::kLittle}) {
    Uint8VectorBuffer write_buf;
    write_buf.set_endian(endian);
    TF_ASSERT_OK(write_buf.Write(kCharValue));
    TF_ASSERT_OK(write_buf.Write(kIntValue));
    TF_ASSERT_OK(write_buf.Write(kBooleanValue));
    TF_ASSERT_OK(write_buf.Write(kColor));
    TF_ASSERT_OK(write_buf.Write(kCharPtrValue));
    TF_ASSERT_OK(write_buf.Write(kStringValue));
    TF_ASSERT_OK(write_buf.Write(kIntBoundedArray));
    TF_ASSERT_OK(write_buf.Write(kIntVector));
    TF_ASSERT_OK(write_buf.Write(kIntArray));
    TF_ASSERT_OK(write_buf.Write(kTuple));

    Buffer read_buf(write_buf.buffer(), write_buf.buffer_len());
    read_buf.set_endian(endian);
    char c;
    TF_ASSERT_OK(read_buf.Read(&c));
    int i;
    TF_ASSERT_OK(read_buf.Read(&i));
    bool b;
    TF_ASSERT_OK(read_buf.Read(&b));
    Color color;
    TF_ASSERT_OK(read_buf.Read(&color));
    std::string s;
    TF_ASSERT_OK(read_buf.Read(&s));
    std::string s2;
    TF_ASSERT_OK(read_buf.Read(&s2));
    uint64_t iba[4];
    TF_ASSERT_OK(read_buf.Read(iba));
    std::vector<int> iv;
    TF_ASSERT_OK(read_buf.Read(&iv));
    std::array<int, 4> ia;
    TF_ASSERT_OK(read_buf.Read(&ia));
    std::tuple<int, int, int> t;
    TF_ASSERT_OK(read_buf.Read(&t));
    ASSERT_TRUE(read_buf.Done());
    EXPECT_EQ(c, kCharValue);
    EXPECT_EQ(i, kIntValue);
    EXPECT_EQ(b, kBooleanValue);
    EXPECT_EQ(color, kColor);
    EXPECT_EQ(s, kCharPtrValue);
    EXPECT_EQ(s2, kStringValue);
    EXPECT_THAT(iba, testing::ElementsAreArray(iba));
    EXPECT_EQ(iv, kIntVector);
    EXPECT_EQ(ia, kIntArray);
    EXPECT_EQ(t, kTuple);
  }
}

TEST(BufferTest, WriteMany) {
  constexpr char kCharValue = 'c';
  constexpr int kIntValue = 12345;
  constexpr bool kBooleanValue = true;
  constexpr Color kColor = Color::kBlue;
  const char* kCharPtrValue = "abc";
  std::string kStringValue = "def";
  uint64_t kIntBoundedArray[4] = {1, 2, 3, 4};
  std::vector<int> kIntVector = {5, 6, 7};
  std::array<int, 4> kIntArray = {8, 9, 10, 11};
  std::tuple<int, int, int> kTuple = std::make_tuple(1, 2, 3);

  for (Endian endian : {Endian::kNative, Endian::kBig, Endian::kLittle}) {
    Uint8VectorBuffer write_buf;
    write_buf.set_endian(endian);
    TF_ASSERT_OK(write_buf.WriteMany(
        kCharValue, kIntValue, kBooleanValue, kColor, kCharPtrValue,
        kStringValue, kIntBoundedArray, kIntVector, kIntArray, kTuple));

    Buffer read_buf(write_buf.buffer(), write_buf.buffer_len());
    read_buf.set_endian(endian);
    char c;
    int i;
    bool b;
    Color color;
    std::string s;
    std::string s2;
    uint64_t iba[4];
    std::vector<int> iv;
    std::array<int, 4> ia;
    std::tuple<int, int, int> t;
    TF_ASSERT_OK(
        read_buf.ReadMany(&c, &i, &b, &color, &s, &s2, iba, &iv, &ia, &t));
    EXPECT_EQ(c, kCharValue);
    EXPECT_EQ(i, kIntValue);
    EXPECT_EQ(b, kBooleanValue);
    EXPECT_EQ(color, kColor);
    EXPECT_EQ(s, kCharPtrValue);
    EXPECT_EQ(s2, kStringValue);
    EXPECT_THAT(iba, testing::ElementsAreArray(iba));
    EXPECT_EQ(iv, kIntVector);
    EXPECT_EQ(ia, kIntArray);
    EXPECT_EQ(t, kTuple);
    ASSERT_TRUE(read_buf.Done());
  }
}

}  // namespace zkx::base
