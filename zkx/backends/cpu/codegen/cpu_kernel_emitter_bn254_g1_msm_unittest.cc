#include "xla/tsl/platform/cpu_info.h"
#include "xla/tsl/platform/status.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/literal_util.h"

namespace zkx::cpu {

TEST_F(CpuKernelEmitterTest, G1MSM) {
  int num_scalar_muls = tsl::port::MaxParallelism() + 1;
  const std::string kHloText = absl::Substitute(R"(
ENTRY %f (x: bn254.sf[$0], y: bn254.g1_affine[$0]) -> bn254.g1_jacobian[] {
  %x = bn254.sf[$0] parameter(0)
  %y = bn254.g1_affine[$0] parameter(1)

  ROOT %ret = bn254.g1_jacobian[] msm(%x, %y)
}
)",
                                                num_scalar_muls);

  TF_ASSERT_OK(Compile(kHloText));

  std::vector<math::bn254::Fr> scalars = base::CreateVector(
      num_scalar_muls, []() { return math::bn254::Fr::Random(); });
  std::vector<math::bn254::G1AffinePoint> bases = base::CreateVector(
      num_scalar_muls, []() { return math::bn254::G1AffinePoint::Random(); });

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(scalars));
  literals.push_back(LiteralUtil::CreateR1<math::bn254::G1AffinePoint>(bases));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  math::bn254::G1JacobianPoint sum;
  for (size_t i = 0; i < scalars.size(); ++i) {
    sum += scalars[i] * bases[i];
  }
  EXPECT_EQ(ret_literal.data<math::bn254::G1JacobianPoint>()[0], sum);
}

}  // namespace zkx::cpu
