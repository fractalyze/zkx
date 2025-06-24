#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/literal_util.h"

namespace zkx::cpu {

TEST_F(CpuKernelEmitterTest, G1ScalarAdd) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.g1_affine[], y: bn254.g1_affine[]) -> bn254.g1_jacobian[]
{
  %x = bn254.g1_affine[] parameter(0)
  %y = bn254.g1_affine[] parameter(1)

  ROOT %ret = bn254.g1_jacobian[] add(%x, %y)
}
)";

  auto x = math::bn254::G1AffinePoint::Random();
  auto y = math::bn254::G1AffinePoint::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::G1AffinePoint>(x);
  Literal y_literal = LiteralUtil::CreateR0<math::bn254::G1AffinePoint>(y);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::G1JacobianPoint>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &y_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  EXPECT_EQ(ret_literal.data<math::bn254::G1JacobianPoint>()[0], x + y);
}

TEST_F(CpuKernelEmitterTest, G1ScalarDouble) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.g1_affine[]) -> bn254.g1_jacobian[] {
  %x = bn254.g1_affine[] parameter(0)

  ROOT %ret = bn254.g1_jacobian[] add(%x, %x)
}
)";

  auto x = math::bn254::G1AffinePoint::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::G1AffinePoint>(x);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::G1JacobianPoint>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  EXPECT_EQ(ret_literal.data<math::bn254::G1JacobianPoint>()[0], x + x);
}

TEST_F(CpuKernelEmitterTest, G1ScalarSub) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.g1_affine[], y: bn254.g1_affine[]) -> bn254.g1_jacobian[]
{
  %x = bn254.g1_affine[] parameter(0)
  %y = bn254.g1_affine[] parameter(1)

  ROOT %ret = bn254.g1_jacobian[] subtract(%x, %y)
}
)";

  auto x = math::bn254::G1AffinePoint::Random();
  auto y = math::bn254::G1AffinePoint::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::G1AffinePoint>(x);
  Literal y_literal = LiteralUtil::CreateR0<math::bn254::G1AffinePoint>(y);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::G1JacobianPoint>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &y_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  EXPECT_EQ(ret_literal.data<math::bn254::G1JacobianPoint>()[0], x - y);
}

TEST_F(CpuKernelEmitterTest, G1ScalarMul) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[], y: bn254.g1_affine[]) -> bn254.g1_jacobian[] {
  %x = bn254.sf[] parameter(0)
  %y = bn254.g1_affine[] parameter(1)

  ROOT %ret = bn254.g1_jacobian[] multiply(%x, %y)
}
)";

  auto x = math::bn254::Fr::Random();
  auto y = math::bn254::G1AffinePoint::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::Fr>(x);
  Literal y_literal = LiteralUtil::CreateR0<math::bn254::G1AffinePoint>(y);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::G1JacobianPoint>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &y_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  EXPECT_EQ(ret_literal.data<math::bn254::G1JacobianPoint>()[0], x * y);
}

TEST_F(CpuKernelEmitterTest, G1MSM) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[4], y: bn254.g1_affine[4]) -> bn254.g1_jacobian[] {
  %x = bn254.sf[4] parameter(0)
  %y = bn254.g1_affine[4] parameter(1)

  ROOT %ret = bn254.g1_jacobian[] msm(%x, %y)
}
)";

  std::vector<math::bn254::Fr> scalars = {
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
  };
  std::vector<math::bn254::G1AffinePoint> bases = {
      math::bn254::G1AffinePoint::Random(),
      math::bn254::G1AffinePoint::Random(),
      math::bn254::G1AffinePoint::Random(),
      math::bn254::G1AffinePoint::Random(),
  };

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
