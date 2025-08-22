#include <iostream>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/strings/substitute.h"

// clang-format off
#include "benchmark/field_flag.h"
#include "benchmark/runner.h"
// clang-format on
#include "zkx/literal_util.h"
#include "zkx/math/elliptic_curves/bn/bn254/fr.h"

namespace zkx::benchmark {

std::string GenerateHlo(Field field, bool inverse, uint32_t degree) {
  size_t size = size_t{1} << degree;
  return absl::Substitute(
      R"(
       ENTRY %f (x: $2[$0]) -> $2[$0] {
         %x = $2[$0] parameter(0)

         ROOT %ret = $2[$0] fft(%x), fft_type=$1, fft_length=$0
       }
       )",
      size, inverse ? "IFFT" : "FFT", FieldToHloString(field));
}

template <typename F>
std::vector<Literal> GenerateLiterals(absl::Span<const F> scalars) {
  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<F>(scalars));
  return literals;
}

template <typename F>
absl::Status RunBenchmark(Runner& runner, Field field, bool inverse) {
  TF_RETURN_IF_ERROR(runner.Run<F>(
      [field, inverse](uint32_t degree) {
        return GenerateHlo(field, inverse, degree);
      },
      [](absl::Span<const F> scalars) { return GenerateLiterals(scalars); }));
  runner.PrintResults();
  return runner.MaybeWriteCsv();
}

absl::Status RealMain(int argc, char** argv) {
  Runner runner;
  Field field;
  bool inverse;

  AddFieldFlag(runner.parser(), &field);
  runner.AddPositionalFlags();
  runner.parser()
      .AddFlag<base::BoolFlag>(&inverse)
      .set_long_name("--inverse")
      .set_default_value(false)
      .set_help("Inverse FFT");
  runner.AddOptionalFlags(Runner::Options{
      .multi_degrees = true,
  });

  TF_RETURN_IF_ERROR(runner.Parse(argc, argv));

  switch (field) {
    case Field::kBn254Fr:
      return RunBenchmark<math::bn254::Fr>(runner, field, inverse);
  }
  ABSL_UNREACHABLE();
  return absl::InternalError("Invalid field");
}

}  // namespace zkx::benchmark

int main(int argc, char* argv[]) {
  absl::Status status = zkx::benchmark::RealMain(argc, argv);
  if (!status.ok()) {
    std::cerr << status.message() << std::endl;
    return 1;
  }
  return 0;
}
