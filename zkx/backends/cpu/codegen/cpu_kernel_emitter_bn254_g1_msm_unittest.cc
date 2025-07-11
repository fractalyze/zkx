#include "absl/log/globals.h"

#include "zkx/base/openmp_util.h"
#include "xla/tsl/platform/cpu_info.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/literal_util.h"

namespace zkx::cpu {

TEST_F(CpuKernelEmitterTest, G1MSM) {
  absl::SetVLogLevel("cpu_kernel_emitter", 4);
  int num_scalar_muls = 1 << 6;
  const std::string kHloText = absl::Substitute(R"(
ENTRY %f (x: bn254.sf[$0], y: bn254.g1_affine[$0]) -> bn254.g1_jacobian[] {
  %x = bn254.sf[$0] parameter(0)
  %y = bn254.g1_affine[$0] parameter(1)

  ROOT %ret = bn254.g1_jacobian[] msm(%x, %y), msm_parallel_type=WINDOW_PARALLEL, msm_pippengers_type=SIGNED_BUCKET_INDEX
}
)",
                                                num_scalar_muls);

  std::vector<math::bn254::Fr> scalars;
  scalars.resize(num_scalar_muls);
  std::vector<math::bn254::G1AffinePoint> bases;
  bases.resize(num_scalar_muls);


  std::vector<math::bn254::G1PointXyzz> bases_xyzz(num_scalar_muls);
  math::bn254::G1PointXyzz g = math::bn254::G1PointXyzz::Generator();
  OMP_PARALLEL_FOR(size_t i = 0; i < num_scalar_muls; ++i) {
    scalars[i] = math::bn254::Fr::Random();
    bases_xyzz[i] = g;
  }
  ASSERT_TRUE(
      math::bn254::G1PointXyzz::BatchToAffine(bases_xyzz, &bases).ok());
  Literal x_label = LiteralUtil::CreateR1<math::bn254::Fr>(scalars);
  Literal y_label = LiteralUtil::CreateR1<math::bn254::G1AffinePoint>(bases);
  Literal ret_label = LiteralUtil::CreateR0<math::bn254::G1JacobianPoint>(0);
  std::vector<Literal*> literals_ptrs = {&x_label, &y_label, &ret_label};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  math::bn254::G1JacobianPoint sum;
  for (size_t i = 0; i < scalars.size(); ++i) {
    sum += scalars[i] * bases[i];
  }
  EXPECT_EQ(ret_label.data<math::bn254::G1JacobianPoint>()[0], sum);
}

}  // namespace zkx::cpu
