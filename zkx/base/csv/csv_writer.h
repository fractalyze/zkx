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

#ifndef ZKX_BASE_CSV_CSV_WRITER_H_
#define ZKX_BASE_CSV_CSV_WRITER_H_

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

#endif  // ZKX_BASE_CSV_CSV_WRITER_H_
