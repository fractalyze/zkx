#include "zkx/base/csv/csv_writer.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace zkx::base {

TEST(CsvWriterTest, WriteSingleRow) {
  CsvWriter writer;
  writer.WriteRow({"a", "b", "c"});
  EXPECT_EQ(writer.ToString(), "a,b,c");
}

TEST(CsvWriterTest, WriteSingleRowWithVector) {
  CsvWriter writer;
  std::vector<std::string> row = {"a", "b", "c"};
  writer.WriteRow(row);
  EXPECT_EQ(writer.ToString(), "a,b,c");
}

TEST(CsvWriterTest, WriteMultipleRows) {
  CsvWriter writer;
  writer.WriteRow({"1", "2", "3"});
  writer.WriteRow({"4", "5", "6"});
  EXPECT_EQ(writer.ToString(), "1,2,3\n4,5,6");
}

TEST(CsvWriterTest, TabSeparator) {
  CsvWriter writer;
  writer.set_separator(CsvSeparator::kTab);
  writer.WriteRow({"x", "y", "z"});
  EXPECT_EQ(writer.ToString(), "x\ty\tz");
}

TEST(CsvWriterTest, SpaceSeparator) {
  CsvWriter writer;
  writer.set_separator(CsvSeparator::kSpace);
  writer.WriteRow({"a", "b", "c"});
  EXPECT_EQ(writer.ToString(), "a b c");
}

TEST(CsvWriterTest, NumericTypes) {
  CsvWriter writer;
  writer.WriteRow({10, 20, 30});
  writer.WriteRow({40, 50, 60});
  EXPECT_EQ(writer.ToString(), "10,20,30\n40,50,60");
}

}  // namespace zkx::base
