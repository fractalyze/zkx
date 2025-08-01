#include "zkx/base/console/table_writer.h"

#include <string>

#include "gtest/gtest.h"

namespace zkx::base {
namespace {

TableWriter CreateTestTable(TableWriterBuilder& builder,
                            int left_whitespace = 0, int right_whitespace = 0) {
  TableWriter writer =
      builder.AddColumn("Name").AddColumn("Age").AddColumn("Country").Build();
  writer.SetElement(0, 0,
                    std::string(left_whitespace, ' ') + "Alice" +
                        std::string(right_whitespace, ' '));
  writer.SetElement(0, 1,
                    std::string(left_whitespace, ' ') + "30" +
                        std::string(right_whitespace, ' '));
  writer.SetElement(0, 2,
                    std::string(left_whitespace, ' ') + "Korea" +
                        std::string(right_whitespace, ' '));
  writer.SetElement(1, 0,
                    std::string(left_whitespace, ' ') + "Bob" +
                        std::string(right_whitespace, ' '));
  writer.SetElement(1, 1,
                    std::string(left_whitespace, ' ') + "7" +
                        std::string(right_whitespace, ' '));
  writer.SetElement(1, 2,
                    std::string(left_whitespace, ' ') + "USA" +
                        std::string(right_whitespace, ' '));
  return writer;
}

}  // namespace

TEST(TableWriterTest, BasicTableOutput) {
  TableWriterBuilder builder;
  TableWriter writer = CreateTestTable(builder);

  std::string output = writer.ToString();
  std::string_view expected =
      "Name AgeCountry\nAlice30 Korea  \nBob  7  USA    ";
  EXPECT_EQ(output, expected);
}

TEST(TableWriterTest, HeaderAlignLeft) {
  TableWriterBuilder builder;
  TableWriter writer = CreateTestTable(builder.AlignHeaderLeft());

  std::string output = writer.ToString();
  std::string_view expected =
      "Name AgeCountry\nAlice30 Korea  \nBob  7  USA    ";
  EXPECT_EQ(output, expected);
}

TEST(TableWriterTest, HeaderAlignRight) {
  TableWriterBuilder builder;
  TableWriter writer = CreateTestTable(builder.AlignHeaderRight());

  std::string output = writer.ToString();
  std::string_view expected =
      " NameAgeCountry\nAlice30 Korea  \nBob  7  USA    ";
  EXPECT_EQ(output, expected);
}

TEST(TableWriterTest, HeaderAlignCenter) {
  TableWriterBuilder builder;
  TableWriter writer = CreateTestTable(builder.AlignHeaderCenter());

  std::string output = writer.ToString();
  std::string_view expected =
      "Name AgeCountry\nAlice30 Korea  \nBob  7  USA    ";
  EXPECT_EQ(output, expected);
}

TEST(TableWriterTest, BodyAlignLeft) {
  TableWriterBuilder builder;
  TableWriter writer = CreateTestTable(builder.AlignBodyLeft());

  std::string output = writer.ToString();
  std::string_view expected =
      "Name AgeCountry\nAlice30 Korea  \nBob  7  USA    ";
  EXPECT_EQ(output, expected);
}

TEST(TableWriterTest, BodyAlignRight) {
  TableWriterBuilder builder;
  TableWriter writer = CreateTestTable(builder.AlignBodyRight());

  std::string output = writer.ToString();
  std::string_view expected =
      "Name AgeCountry\nAlice 30  Korea\n  Bob  7    USA";
  EXPECT_EQ(output, expected);
}

TEST(TableWriterTest, BodyAlignCenter) {
  TableWriterBuilder builder;
  TableWriter writer = CreateTestTable(builder.AlignBodyCenter());

  std::string output = writer.ToString();
  std::string_view expected =
      "Name AgeCountry\nAlice30  Korea \n Bob  7   USA  ";
  EXPECT_EQ(output, expected);
}

TEST(TableWriterTest, AddSpace) {
  TableWriterBuilder builder;
  TableWriter writer = CreateTestTable(builder.AddSpace(2));

  std::string output = writer.ToString();
  std::string_view expected =
      "Name   Age  Country\nAlice  30   Korea  \nBob    7    USA    ";
  EXPECT_EQ(output, expected);
}

TEST(TableWriterTest, StripBothAsciiWhitespace) {
  TableWriterBuilder builder;
  TableWriter writer =
      CreateTestTable(builder.StripBothAsciiWhitespace(), 2, 2);

  std::string output = writer.ToString();
  std::string_view expected =
      "Name AgeCountry\nAlice30 Korea  \nBob  7  USA    ";
  EXPECT_EQ(output, expected);
}

TEST(TableWriterTest, StripTrailingAsciiWhitespace) {
  TableWriterBuilder builder;
  TableWriter writer =
      CreateTestTable(builder.StripTrailingAsciiWhitespace(), 0, 2);

  std::string output = writer.ToString();
  std::string_view expected =
      "Name AgeCountry\nAlice30 Korea  \nBob  7  USA    ";
  EXPECT_EQ(output, expected);
}

TEST(TableWriterTest, StripLeadingAsciiWhitespace) {
  TableWriterBuilder builder;
  TableWriter writer =
      CreateTestTable(builder.StripLeadingAsciiWhitespace(), 2, 0);

  std::string output = writer.ToString();
  std::string_view expected =
      "Name AgeCountry\nAlice30 Korea  \nBob  7  USA    ";
  EXPECT_EQ(output, expected);
}

}  // namespace zkx::base
