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

  auto x = math::bn254::G2AffinePoint::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(x);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::G2PointXyzz>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

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

  auto x = math::bn254::G2AffinePoint::Random();
  auto y = math::bn254::G2AffinePoint::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(x);
  Literal y_literal = LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(y);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::G2JacobianPoint>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &y_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  EXPECT_EQ(ret_literal.data<math::bn254::G2JacobianPoint>()[0], x + y);
}

TEST_F(CpuKernelEmitterTest, G2ScalarDouble) {
  const std::string kHloText = R"(
  ENTRY %f (x: bn254.g2_affine[]) -> bn254.g2_jacobian[] {
    %x = bn254.g2_affine[] parameter(0)

    ROOT %ret = bn254.g2_jacobian[] add(%x, %x)
  }
  )";

  auto x = math::bn254::G2AffinePoint::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(x);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::G2JacobianPoint>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

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

  auto x = math::bn254::G2AffinePoint::Random();
  auto y = math::bn254::G2AffinePoint::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(x);
  Literal y_literal = LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(y);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::G2JacobianPoint>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &y_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

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

  auto x = math::bn254::Fr::Random();
  auto y = math::bn254::G2AffinePoint::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::Fr>(x);
  Literal y_literal = LiteralUtil::CreateR0<math::bn254::G2AffinePoint>(y);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::G2JacobianPoint>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &y_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  EXPECT_EQ(ret_literal.data<math::bn254::G2JacobianPoint>()[0], x * y);
}

}  // namespace zkx::cpu
