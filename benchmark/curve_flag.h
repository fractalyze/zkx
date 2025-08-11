#ifndef BENCHMARK_CURVE_FLAG_H_
#define BENCHMARK_CURVE_FLAG_H_

#include <string_view>

#include "absl/status/status.h"

#include "zkx/base/flag/flag_parser.h"
#include "zkx/base/flag/flag_value_traits.h"

namespace zkx {
namespace benchmark {

enum class Curve {
  kBn254G1,
  kBn254G2,
};

void AddCurveFlag(base::FlagParserBase& parser, benchmark::Curve* curve);

std::string_view CurveToScalarFieldHloString(Curve curve);

std::string_view CurveToHloString(Curve curve);

}  // namespace benchmark

namespace base {

template <>
class FlagValueTraits<benchmark::Curve> {
 public:
  static absl::Status ParseValue(std::string_view input,
                                 benchmark::Curve* value);
};

}  // namespace base
}  // namespace zkx

#endif  // BENCHMARK_CURVE_FLAG_H_
