#ifndef ZX_BASE_CSV_CSV_WRITER_H_
#define ZX_BASE_CSV_CSV_WRITER_H_

#include <sstream>
#include <string_view>

#include "absl/types/span.h"

#include "xla/tsl/platform/env.h"
#include "zkx/base/csv/csv_separator.h"

namespace zkx::base {

class CsvWriter {
 public:
  CsvWriter() = default;
  CsvWriter(const CsvWriter&) = delete;
  CsvWriter& operator=(const CsvWriter&) = delete;

  void set_separator(CsvSeparator separator) { separator_ = separator; }

  template <typename T>
  void WriteRow(std::initializer_list<T> row) {
    WriteRowImpl(row.begin(), row.end());
  }

  template <typename Row>
  void WriteRow(const Row& row) {
    WriteRowImpl(std::begin(row), std::end(row));
  }

  std::string ToString() const { return ss_.str(); }

  absl::Status WriteToFile(std::string_view path) const {
    return tsl::WriteStringToFile(tsl::Env::Default(), path, ToString());
  }

 private:
  template <typename Iter>
  void WriteRowImpl(Iter begin, Iter end) {
    if (is_first_) {
      is_first_ = false;
    } else {
      ss_ << '\n';
    }
    for (Iter it = begin; it != end; ++it) {
      if (it != begin) ss_ << static_cast<char>(separator_);
      ss_ << *it;
    }
  }

  bool is_first_ = true;
  CsvSeparator separator_ = CsvSeparator::kComma;

  std::stringstream ss_;
};

}  // namespace zkx::base

#endif  // ZX_BASE_CSV_CSV_WRITER_H_
