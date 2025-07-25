#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/literal_util.h"

namespace zkx::cpu {

TEST_F(CpuKernelEmitterTest, G2ScalarConvert) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.g2_affine[]) -> bn254.g2_xyzz[]
{
  %x = bn254.g2_affine[] parameter(0)

  ROOT %ret = bn254.g2_xyzz[] convert(%x)
}
)";

  Compile(kHloText);

  auto x = math::bn254::G2AffinePoint::Random();

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(x));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  EXPECT_EQ(ret_literal.data<math::bn254::G2PointXyzz>()[0], x.ToXyzz());
}

TEST_F(CpuKernelEmitterTest, G2ScalarAdd) {
  const std::string kHloText = R"(
  ENTRY %f (x: bn254.g2_affine[], y: bn254.g2_affine[]) -> bn254.g2_jacobian[] {
    %x = bn254.g2_affine[] parameter(0)
    %y = bn254.g2_affine[] parameter(1)

    ROOT %ret = bn254.g2_jacobian[] add(%x, %y)
  }
  )";

  Compile(kHloText);

  auto x = math::bn254::G2AffinePoint::Random();
  auto y = math::bn254::G2AffinePoint::Random();

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(x));
  literals.push_back(LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(y));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  EXPECT_EQ(ret_literal.data<math::bn254::G2JacobianPoint>()[0], x + y);
}

TEST_F(CpuKernelEmitterTest, G2ScalarDouble) {
  const std::string kHloText = R"(
  ENTRY %f (x: bn254.g2_affine[]) -> bn254.g2_jacobian[] {
    %x = bn254.g2_affine[] parameter(0)

    ROOT %ret = bn254.g2_jacobian[] add(%x, %x)
  }
  )";

  Compile(kHloText);

  auto x = math::bn254::G2AffinePoint::Random();

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(x));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  EXPECT_EQ(ret_literal.data<math::bn254::G2JacobianPoint>()[0], x + x);
}

TEST_F(CpuKernelEmitterTest, G2ScalarSub) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.g2_affine[], y: bn254.g2_affine[]) -> bn254.g2_jacobian[] {
  %x = bn254.g2_affine[] parameter(0)
  %y = bn254.g2_affine[] parameter(1)

  ROOT %ret = bn254.g2_jacobian[] subtract(%x, %y)
}
)";

  Compile(kHloText);

  auto x = math::bn254::G2AffinePoint::Random();
  auto y = math::bn254::G2AffinePoint::Random();

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(x));
  literals.push_back(LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(y));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  EXPECT_EQ(ret_literal.data<math::bn254::G2JacobianPoint>()[0], x - y);
}

TEST_F(CpuKernelEmitterTest, G2ScalarMul) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[], y: bn254.g2_affine[]) -> bn254.g2_jacobian[] {
  %x = bn254.sf[] parameter(0)
  %y = bn254.g2_affine[] parameter(1)

  ROOT %ret = bn254.g2_jacobian[] multiply(%x, %y)
}
)";

  Compile(kHloText);

  auto x = math::bn254::Fr::Random();
  auto y = math::bn254::G2AffinePoint::Random();

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(x));
  literals.push_back(LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(y));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  EXPECT_EQ(ret_literal.data<math::bn254::G2JacobianPoint>()[0], x * y);
}

}  // namespace zkx::cpu
