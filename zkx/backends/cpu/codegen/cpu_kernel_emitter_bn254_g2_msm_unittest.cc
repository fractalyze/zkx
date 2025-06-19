#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/literal_util.h"

namespace zkx::cpu {

TEST_F(CpuKernelEmitterTest, G2MSM) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[4], y: bn254.g2_affine[4]) -> bn254.g2_jacobian[] {
  %x = bn254.sf[4] parameter(0)
  %y = bn254.g2_affine[4] parameter(1)

  ROOT %ret = bn254.g2_jacobian[] msm(%x, %y)
}
)";

  std::vector<math::bn254::Fr> scalars = {
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
  };
  std::vector<math::bn254::G2AffinePoint> bases = {
      math::bn254::G2AffinePoint::Random(),
      math::bn254::G2AffinePoint::Random(),
      math::bn254::G2AffinePoint::Random(),
      math::bn254::G2AffinePoint::Random(),
  };

  Literal x_literal = LiteralUtil::CreateR1<math::bn254::Fr>(scalars);
  Literal y_literal = LiteralUtil::CreateR1<math::bn254::G2AffinePoint>(bases);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::G2JacobianPoint>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &y_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  math::bn254::G2JacobianPoint sum;
  for (size_t i = 0; i < scalars.size(); ++i) {
    sum += scalars[i] * bases[i];
  }
  EXPECT_EQ(ret_literal.data<math::bn254::G2JacobianPoint>()[0], sum);
}

}  // namespace zkx::cpu
