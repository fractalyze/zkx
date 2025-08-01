#include "benchmark/field_flag.h"

#include "absl/log/check.h"
#include "absl/strings/substitute.h"

#include "zkx/base/flag/flag.h"

namespace zkx {
namespace benchmark {

void AddFieldFlag(base::FlagParserBase& parser, benchmark::Field* field) {
  parser.AddFlag<base::Flag<benchmark::Field>>(field)
      .set_name("field")
      .set_help("Field to generate scalars for");
}

std::string_view FieldToHloString(Field field) {
  switch (field) {
    case Field::kBn254Fr:
      return "bn254.sf";
  }
  ABSL_UNREACHABLE();
  return "";
}

}  // namespace benchmark

namespace base {

// static
absl::Status FlagValueTraits<benchmark::Field>::ParseValue(
    std::string_view input, benchmark::Field* value) {
  if (input == "bn254-fr") {
    *value = benchmark::Field::kBn254Fr;
  } else {
    return absl::NotFoundError(absl::Substitute("Unknown field: $0", input));
  }
  return absl::OkStatus();
}

}  // namespace base
}  // namespace zkx
