#ifndef BENCHMARK_FIELD_FLAG_H_
#define BENCHMARK_FIELD_FLAG_H_

#include "absl/status/status.h"

#include "zkx/base/flag/flag_parser.h"
#include "zkx/base/flag/flag_value_traits.h"

namespace zkx {
namespace benchmark {

enum class Field {
  kBn254Fr,
};

void AddFieldFlag(base::FlagParserBase& parser, benchmark::Field* field);

std::string_view FieldToHloString(Field field);

}  // namespace benchmark

namespace base {

template <>
class FlagValueTraits<benchmark::Field> {
 public:
  static absl::Status ParseValue(std::string_view input,
                                 benchmark::Field* value);
};

}  // namespace base
}  // namespace zkx

#endif  // BENCHMARK_FIELD_FLAG_H_
