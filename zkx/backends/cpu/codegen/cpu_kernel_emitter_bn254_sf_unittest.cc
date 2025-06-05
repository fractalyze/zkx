#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/literal_util.h"
#include "zkx/math/poly/root_of_unity.h"

namespace zkx::cpu {

TEST_F(CpuKernelEmitterTest, FieldScalarAdd) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[], y: bn254.sf[]) -> bn254.sf[] {
  %x = bn254.sf[] parameter(0)
  %y = bn254.sf[] parameter(1)

  ROOT %ret = bn254.sf[] add(%x, %y)
}
)";

  auto x = math::bn254::Fr::Random();
  auto y = math::bn254::Fr::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::Fr>(x);
  Literal y_literal = LiteralUtil::CreateR0<math::bn254::Fr>(y);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::Fr>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &y_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  EXPECT_EQ(ret_literal, LiteralUtil::CreateR0<math::bn254::Fr>(x + y));
}

TEST_F(CpuKernelEmitterTest, FieldScalarSub) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[], y: bn254.sf[]) -> bn254.sf[] {
  %x = bn254.sf[] parameter(0)
  %y = bn254.sf[] parameter(1)

  ROOT %ret = bn254.sf[] subtract(%x, %y)
}
)";

  auto x = math::bn254::Fr::Random();
  auto y = math::bn254::Fr::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::Fr>(x);
  Literal y_literal = LiteralUtil::CreateR0<math::bn254::Fr>(y);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::Fr>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &y_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  EXPECT_EQ(ret_literal, LiteralUtil::CreateR0<math::bn254::Fr>(x - y));
}

TEST_F(CpuKernelEmitterTest, FieldScalarMul) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[], y: bn254.sf[]) -> bn254.sf[] {
  %x = bn254.sf[] parameter(0)
  %y = bn254.sf[] parameter(1)

  ROOT %ret = bn254.sf[] multiply(%x, %y)
}
)";

  auto x = math::bn254::Fr::Random();
  auto y = math::bn254::Fr::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::Fr>(x);
  Literal y_literal = LiteralUtil::CreateR0<math::bn254::Fr>(y);
  Literal ret_literal = LiteralUtil::CreateR0<math::bn254::Fr>(0);
  std::vector<Literal*> literals_ptrs = {&x_literal, &y_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  EXPECT_EQ(ret_literal, LiteralUtil::CreateR0<math::bn254::Fr>(x * y));
}

TEST_F(CpuKernelEmitterTest, FieldTensorAdd) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[2,3], y: bn254.sf[2,3]) -> bn254.sf[2,3] {
  %x = bn254.sf[2,3] parameter(0)
  %y = bn254.sf[2,3] parameter(1)

  ROOT %ret = bn254.sf[2,3] add(%x, %y)
}
)";

  std::vector<std::vector<math::bn254::Fr>> x = {{
                                                     math::bn254::Fr::Random(),
                                                     math::bn254::Fr::Random(),
                                                     math::bn254::Fr::Random(),
                                                 },
                                                 {
                                                     math::bn254::Fr::Random(),
                                                     math::bn254::Fr::Random(),
                                                     math::bn254::Fr::Random(),
                                                 }};
  std::vector<std::vector<math::bn254::Fr>> y = {{
                                                     math::bn254::Fr::Random(),
                                                     math::bn254::Fr::Random(),
                                                     math::bn254::Fr::Random(),
                                                 },
                                                 {
                                                     math::bn254::Fr::Random(),
                                                     math::bn254::Fr::Random(),
                                                     math::bn254::Fr::Random(),
                                                 }};

  Literal x_literal = LiteralUtil::CreateR2<math::bn254::Fr>({
      {x[0][0], x[0][1], x[0][2]},
      {x[1][0], x[1][1], x[1][2]},
  });
  Literal y_literal = LiteralUtil::CreateR2<math::bn254::Fr>({
      {y[0][0], y[0][1], y[0][2]},
      {y[1][0], y[1][1], y[1][2]},
  });
  Literal ret_literal = LiteralUtil::CreateR2<math::bn254::Fr>({
      {0, 0, 0},
      {0, 0, 0},
  });
  std::vector<Literal*> literals_ptrs = {&x_literal, &y_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  for (int64_t i = 0; i < x.size(); ++i) {
    for (int64_t j = 0; j < x[i].size(); ++j) {
      EXPECT_EQ(ret_literal.Get<math::bn254::Fr>({i, j}), x[i][j] + y[i][j]);
    }
  }
}

TEST_F(CpuKernelEmitterTest, FFT) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[4]) -> bn254.sf[4] {
  %x = bn254.sf[4] parameter(0)

  ROOT %ret = bn254.sf[4] fft(%x), fft_type=FFT, fft_length=4
}
)";

  std::vector<math::bn254::Fr> coeffs = {
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
  };

  Literal literal =
      LiteralUtil::CreateR1<math::bn254::Fr>(absl::MakeSpan(coeffs));
  std::vector<Literal*> literals_ptrs = {&literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  TF_ASSERT_OK_AND_ASSIGN(math::bn254::Fr w,
                          math::GetRootOfUnity<math::bn254::Fr>(coeffs.size()));
  std::vector<math::bn254::Fr> twiddles(coeffs.size());
  for (int64_t i = 0; i < coeffs.size(); ++i) {
    twiddles[i] = w.Pow(i);
  }
  for (int64_t i = 0; i < coeffs.size(); ++i) {
    auto expected = math::bn254::Fr::Zero();
    for (int64_t j = 0; j < coeffs.size(); ++j) {
      expected += coeffs[j] * twiddles[(i * j) % coeffs.size()];
    }
    EXPECT_EQ(literal.Get<math::bn254::Fr>({i}), expected);
  }
}

TEST_F(CpuKernelEmitterTest, IFFT) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[4]) -> bn254.sf[4] {
  %x = bn254.sf[4] parameter(0)
  ROOT %ret = bn254.sf[4] fft(%x), fft_type=IFFT, fft_length=4
}
)";

  std::vector<math::bn254::Fr> evals = {
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
  };

  Literal literal =
      LiteralUtil::CreateR1<math::bn254::Fr>(absl::MakeSpan(evals));
  std::vector<Literal*> literals_ptrs = {&literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  TF_ASSERT_OK_AND_ASSIGN(math::bn254::Fr w,
                          math::GetRootOfUnity<math::bn254::Fr>(evals.size()));
  math::bn254::Fr n_inv = *math::bn254::Fr(evals.size()).Inverse();
  std::vector<math::bn254::Fr> twiddles(evals.size());
  for (int64_t i = 0; i < evals.size(); ++i) {
    twiddles[i] = *w.Pow(i).Inverse();
  }

  for (int64_t i = 0; i < evals.size(); ++i) {
    auto expected = math::bn254::Fr::Zero();
    for (int64_t j = 0; j < evals.size(); ++j) {
      expected += evals[j] * twiddles[(i * j) % evals.size()];
    }
    expected = expected * n_inv;
    EXPECT_EQ(literal.Get<math::bn254::Fr>({i}), expected);
  }
}

TEST_F(CpuKernelEmitterTest, BroadcastScalar) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[]) -> bn254.sf[4] {
  %x = bn254.sf[] parameter(0)

  ROOT %ret = bn254.sf[4] broadcast(%x)
}
)";

  auto x = math::bn254::Fr::Random();

  Literal x_literal = LiteralUtil::CreateR0<math::bn254::Fr>(x);
  Literal ret_literal = LiteralUtil::CreateR1<math::bn254::Fr>({0, 0, 0, 0});
  std::vector<Literal*> literals_ptrs = {&x_literal, &ret_literal};
  RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

  EXPECT_EQ(ret_literal, LiteralUtil::CreateR1<math::bn254::Fr>({x, x, x, x}));
}

TEST_F(CpuKernelEmitterTest, BroadcastTensor) {
  std::vector<math::bn254::Fr> x = {
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
  };
  Literal kAnswers[] = {
      LiteralUtil::CreateR3<math::bn254::Fr>(
          {{{x[0], x[0]}, {x[0], x[0]}}, {{x[1], x[1]}, {x[1], x[1]}}}),
      LiteralUtil::CreateR3<math::bn254::Fr>(
          {{{x[0], x[0]}, {x[1], x[1]}}, {{x[0], x[0]}, {x[1], x[1]}}}),
      LiteralUtil::CreateR3<math::bn254::Fr>(
          {{{x[0], x[1]}, {x[0], x[1]}}, {{x[0], x[1]}, {x[0], x[1]}}}),
  };

  for (size_t i = 0; i < std::size(kAnswers); ++i) {
    const std::string kHloText = absl::Substitute(R"(
  ENTRY %f (x: bn254.sf[2]) -> bn254.sf[2, 2, 2] {
    %x = bn254.sf[2] parameter(0)

    ROOT %ret = bn254.sf[2, 2, 2] broadcast(%x), dimensions={$0}
  }
  )",
                                                  i);

    Literal x_literal =
        LiteralUtil::CreateR1<math::bn254::Fr>(absl::MakeSpan(x));
    Literal ret_literal = LiteralUtil::CreateR3<math::bn254::Fr>(
        {{{0, 0}, {0, 0}}, {{0, 0}, {0, 0}}});
    std::vector<Literal*> literals_ptrs = {&x_literal, &ret_literal};
    RunHlo(kHloText, absl::MakeSpan(literals_ptrs));

    EXPECT_EQ(kAnswers[i], ret_literal);
  }
}

}  // namespace zkx::cpu
