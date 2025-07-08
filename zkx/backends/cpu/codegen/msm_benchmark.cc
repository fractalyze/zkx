#include "benchmark/benchmark.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/base/openmp_util.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/literal_util.h"
#include "zkx/math/elliptic_curves/bn/bn254/g1.h"
#include "zkx/service/hlo_runner.h"
#include "zkx/service/platform_util.h"

namespace zkx::math {

struct TestInput {
 public:
  static absl::StatusOr<TestInput> Create(size_t num_scalar_muls) {
    TestInput test_input;
    test_input.x.resize(num_scalar_muls);
    test_input.y.resize(num_scalar_muls);
    std::vector<math::bn254::G1PointXyzz> y_xyzz(num_scalar_muls);
    math::bn254::G1PointXyzz g = math::bn254::G1PointXyzz::Generator();
    OMP_PARALLEL_FOR(size_t i = 0; i < num_scalar_muls; ++i) {
      test_input.x[i] = math::bn254::Fr::Random();
      y_xyzz[i] = g;
    }
    TF_RETURN_IF_ERROR(
        math::bn254::G1PointXyzz::BatchToAffine(y_xyzz, &test_input.y));
    return test_input;
  }

  std::vector<math::bn254::Fr> x;
  std::vector<math::bn254::G1AffinePoint> y;
};

template <typename Point, MsmParallelType ParallelType>
absl::Status RunPippenger(benchmark::State& state) {
  size_t num_scalar_muls = state.range(0);

  const std::string kHloText = absl::Substitute(
      R"(
    ENTRY %f (x: bn254.sf[$0], y: bn254.g1_affine[$0]) -> bn254.g1_xyzz[] {
      %x = bn254.sf[$0] parameter(0)
      %y = bn254.g1_affine[$0] parameter(1)

      ROOT %ret = bn254.g1_xyzz[] msm(%x, %y), msm_parallel_type=$1
    }
    )",
      num_scalar_muls,
      ParallelType == MsmParallelType::WINDOW_PARALLEL ? "WINDOW_PARALLEL"
                                                       : "TERM_PARALLEL");

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnUnverifiedModule(kHloText));
  HloRunner runner(PlatformUtil::GetPlatform("cpu").value());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<OpaqueExecutable> opaque_executable,
                      runner.CreateExecutable(std::move(module),
                                              /*run_hlo_passes=*/false));

  TF_ASSIGN_OR_RETURN(TestInput test_input, TestInput::Create(num_scalar_muls));

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(test_input.x));
  literals.push_back(
      LiteralUtil::CreateR1<math::bn254::G1AffinePoint>(test_input.y));

  for (auto _ : state) {
    TF_ASSIGN_OR_RETURN(Literal output, runner.ExecuteWithExecutable(
                                            opaque_executable.get(), literals,
                                            /*profile=*/nullptr));
    benchmark::DoNotOptimize(output);
  }
  return absl::OkStatus();
}

template <typename Point, MsmParallelType ParallelType>
void BM_Pippenger(benchmark::State& state) {
  absl::Status status = RunPippenger<Point, ParallelType>(state);
  CHECK(status.ok());
}

BENCHMARK_TEMPLATE(BM_Pippenger, bn254::G1AffinePoint,
                   MsmParallelType::WINDOW_PARALLEL)
    ->Unit(::benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1 << 18, 1 << 22);

BENCHMARK_TEMPLATE(BM_Pippenger, bn254::G1AffinePoint,
                   MsmParallelType::TERM_PARALLEL)
    ->Unit(::benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1 << 18, 1 << 22);

}  // namespace zkx::math
