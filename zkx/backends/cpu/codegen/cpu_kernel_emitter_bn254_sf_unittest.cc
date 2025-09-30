#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/literal_util.h"
#include "zkx/math/base/sparse_matrix.h"
#include "zkx/math/poly/root_of_unity.h"

namespace zkx::cpu {

TEST_F(CpuKernelEmitterTest, FieldScalarConvert) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[]) -> bn254.sf_std[] {
  %x = bn254.sf[] parameter(0)

  ROOT %ret = bn254.sf_std[] convert(%x)
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  auto x = math::bn254::Fr::Random();

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(x));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  EXPECT_EQ(ret_literal,
            LiteralUtil::CreateR0<math::bn254::FrStd>(x.MontReduce()));
}

TEST_F(CpuKernelEmitterTest, FieldScalarNegate) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[]) -> bn254.sf[] {
  %x = bn254.sf[] parameter(0)

  ROOT %ret = bn254.sf[] negate(%x)
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  auto x = math::bn254::Fr::Random();

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(x));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  EXPECT_EQ(ret_literal, LiteralUtil::CreateR0<math::bn254::Fr>(-x));
}

TEST_F(CpuKernelEmitterTest, FieldScalarInverse) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[]) -> bn254.sf[] {
  %x = bn254.sf[] parameter(0)

  ROOT %ret = bn254.sf[] inverse(%x)
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  auto x = math::bn254::Fr::Random();
  while (x.IsZero()) {
    x = math::bn254::Fr::Random();
  }

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(x));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  EXPECT_EQ(ret_literal, LiteralUtil::CreateR0<math::bn254::Fr>(*x.Inverse()));
}

TEST_F(CpuKernelEmitterTest, FieldScalarBatchInverse) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[4]) -> bn254.sf[4] {
  %x = bn254.sf[4] parameter(0)

  ROOT %ret = bn254.sf[4] inverse(%x)
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  std::vector<math::bn254::Fr> x(4, math::bn254::Fr::Zero());
  for (size_t i = 0; i < x.size(); ++i) {
    while (x[i].IsZero()) {
      x[i] = math::bn254::Fr::Random();
    }
  }

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(x));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  std::vector<math::bn254::Fr> expected;
  TF_ASSERT_OK(math::BatchInverse(x, &expected));
  EXPECT_EQ(ret_literal, LiteralUtil::CreateR1<math::bn254::Fr>(expected));
}

TEST_F(CpuKernelEmitterTest, FieldScalarAdd) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[], y: bn254.sf[]) -> bn254.sf[] {
  %x = bn254.sf[] parameter(0)
  %y = bn254.sf[] parameter(1)

  ROOT %ret = bn254.sf[] add(%x, %y)
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  auto x = math::bn254::Fr::Random();
  auto y = math::bn254::Fr::Random();
  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(x));
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(y));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

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

  TF_ASSERT_OK(Compile(kHloText));

  auto x = math::bn254::Fr::Random();
  auto y = math::bn254::Fr::Random();

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(x));
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(y));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

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

  TF_ASSERT_OK(Compile(kHloText));

  auto x = math::bn254::Fr::Random();
  auto y = math::bn254::Fr::Random();

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(x));
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(y));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  EXPECT_EQ(ret_literal, LiteralUtil::CreateR0<math::bn254::Fr>(x * y));
}

TEST_F(CpuKernelEmitterTest, FieldScalarDiv) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[], y: bn254.sf[]) -> bn254.sf[] {
  %x = bn254.sf[] parameter(0)
  %y = bn254.sf[] parameter(1)

  ROOT %ret = bn254.sf[] divide(%x, %y)
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  auto x = math::bn254::Fr::Random();
  auto y = math::bn254::Fr::Random();
  while (y.IsZero()) {
    y = math::bn254::Fr::Random();
  }

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(x));
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(y));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  EXPECT_EQ(ret_literal, LiteralUtil::CreateR0<math::bn254::Fr>(*(x / y)));
}

TEST_F(CpuKernelEmitterTest, FieldScalarPow) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[], y: u32[]) -> bn254.sf[] {
  %x = bn254.sf[] parameter(0)
  %y = u32[] parameter(1)

  ROOT %ret = bn254.sf[] power(%x, %y)
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  auto x = math::bn254::Fr::Random();
  auto y = base::Uniform<uint32_t>();

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(x));
  literals.push_back(LiteralUtil::CreateR0<uint32_t>(y));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  EXPECT_EQ(ret_literal, LiteralUtil::CreateR0<math::bn254::Fr>(x.Pow(y)));
}

TEST_F(CpuKernelEmitterTest, FieldScalarPowWithSignedExponentShouldFail) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[], y: s32[]) -> bn254.sf[] {
  %x = bn254.sf[] parameter(0)
  %y = s32[] parameter(1)

  ROOT %ret = bn254.sf[] power(%x, %y)
}
)";

  EXPECT_THAT(Compile(kHloText),
              ::tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CpuKernelEmitterTest, FieldTensorAdd) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[2,3], y: bn254.sf[2,3]) -> bn254.sf[2,3] {
  %x = bn254.sf[2,3] parameter(0)
  %y = bn254.sf[2,3] parameter(1)

  ROOT %ret = bn254.sf[2,3] add(%x, %y)
}
)";

  TF_ASSERT_OK(Compile(kHloText));

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

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR2<math::bn254::Fr>({
      {x[0][0], x[0][1], x[0][2]},
      {x[1][0], x[1][1], x[1][2]},
  }));
  literals.push_back(LiteralUtil::CreateR2<math::bn254::Fr>({
      {y[0][0], y[0][1], y[0][2]},
      {y[1][0], y[1][1], y[1][2]},
  }));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

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

  TF_ASSERT_OK(Compile(kHloText));

  std::vector<math::bn254::Fr> coeffs = {
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
  };

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(coeffs));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

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
    EXPECT_EQ(ret_literal.Get<math::bn254::Fr>({i}), expected);
  }
}

TEST_F(CpuKernelEmitterTest, FFTWithTwiddles) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[4], twiddles: bn254.sf[4]) -> bn254.sf[4] {
  %x = bn254.sf[4] parameter(0)
  %twiddles = bn254.sf[4] parameter(1)

  ROOT %ret = bn254.sf[4] fft(%x, %twiddles), fft_type=FFT, fft_length=4
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  std::vector<math::bn254::Fr> coeffs = {
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
  };

  TF_ASSERT_OK_AND_ASSIGN(math::bn254::Fr w,
                          math::GetRootOfUnity<math::bn254::Fr>(coeffs.size()));
  std::vector<math::bn254::Fr> twiddles(coeffs.size());
  for (int64_t i = 0; i < coeffs.size(); ++i) {
    twiddles[i] = w.Pow(i);
  }

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(coeffs));
  literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(twiddles));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  for (int64_t i = 0; i < coeffs.size(); ++i) {
    auto expected = math::bn254::Fr::Zero();
    for (int64_t j = 0; j < coeffs.size(); ++j) {
      expected += coeffs[j] * twiddles[(i * j) % coeffs.size()];
    }
    EXPECT_EQ(ret_literal.Get<math::bn254::Fr>({i}), expected);
  }
}

TEST_F(CpuKernelEmitterTest, IFFT) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[4]) -> bn254.sf[4] {
  %x = bn254.sf[4] parameter(0)
  ROOT %ret = bn254.sf[4] fft(%x), fft_type=IFFT, fft_length=4
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  std::vector<math::bn254::Fr> evals = {
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
  };

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(evals));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

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
    EXPECT_EQ(ret_literal.Get<math::bn254::Fr>({i}), expected);
  }
}

TEST_F(CpuKernelEmitterTest, IFFTWithTwiddles) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[4], twiddles: bn254.sf[4]) -> bn254.sf[4] {
  %x = bn254.sf[4] parameter(0)
  %twiddles = bn254.sf[4] parameter(1)

  ROOT %ret = bn254.sf[4] fft(%x, %twiddles), fft_type=IFFT, fft_length=4
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  std::vector<math::bn254::Fr> evals = {
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
      math::bn254::Fr::Random(),
  };

  TF_ASSERT_OK_AND_ASSIGN(math::bn254::Fr w,
                          math::GetRootOfUnity<math::bn254::Fr>(evals.size()));
  math::bn254::Fr n_inv = *math::bn254::Fr(evals.size()).Inverse();
  std::vector<math::bn254::Fr> twiddles(evals.size());
  for (int64_t i = 0; i < evals.size(); ++i) {
    twiddles[i] = *w.Pow(i).Inverse();
  }

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(evals));
  literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(twiddles));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  for (int64_t i = 0; i < evals.size(); ++i) {
    auto expected = math::bn254::Fr::Zero();
    for (int64_t j = 0; j < evals.size(); ++j) {
      expected += evals[j] * twiddles[(i * j) % evals.size()];
    }
    expected = expected * n_inv;
    EXPECT_EQ(ret_literal.Get<math::bn254::Fr>({i}), expected);
  }
}

TEST_F(CpuKernelEmitterTest, BroadcastScalar) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[]) -> bn254.sf[4] {
  %x = bn254.sf[] parameter(0)

  ROOT %ret = bn254.sf[4] broadcast(%x)
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  auto x = math::bn254::Fr::Random();

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR0<math::bn254::Fr>(x));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

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

    TF_ASSERT_OK(Compile(kHloText));

    std::vector<Literal> literals;
    literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(x));
    TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

    EXPECT_EQ(ret_literal, kAnswers[i]);
  }
}

TEST_F(CpuKernelEmitterTest, FieldCSRMatrixVectorMultiplication) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[4,3]{1,0:D(D, C)NNZ(8)}, y: bn254.sf[3]) -> bn254.sf[4] {
  %x = bn254.sf[4,3]{1,0:D(D, C)NNZ(8)} parameter(0)
  %y = bn254.sf[3] parameter(1)

  ROOT %ret = bn254.sf[4] dot(%x, %y)
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  math::SparseMatrix<math::bn254::Fr> x =
      math::SparseMatrix<math::bn254::Fr>::Random(4, 3, 8);

  std::vector<uint32_t> x_row_ptrs, x_col_indices;
  std::vector<math::bn254::Fr> x_values;
  x.ToCSR(x_row_ptrs, x_col_indices, x_values);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> x_buffer, x.ToCSRBuffer());
  std::array<math::bn254::Fr, 3> y = {math::bn254::Fr::Random(),
                                      math::bn254::Fr::Random(),
                                      math::bn254::Fr::Random()};

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<uint8_t>(x_buffer));
  literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(y));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  std::vector<math::bn254::Fr> expected(x_row_ptrs.size() - 1);
  for (int64_t i = 0; i < expected.size(); ++i) {
    for (int64_t j = x_row_ptrs[i]; j < x_row_ptrs[i + 1]; ++j) {
      expected[i] += x_values[j] * y[x_col_indices[j]];
    }
  }

  for (int64_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(ret_literal.Get<math::bn254::Fr>({i}), expected[i]);
  }
}

TEST_F(CpuKernelEmitterTest, Slice) {
  const std::string kHloText = R"(
ENTRY %f (x: bn254.sf[]) -> bn254.sf[3] {
  %x = bn254.sf[6] parameter(0)

  ROOT ret = bn254.sf[3] slice(%x), slice={[2:5]}
}
)";

  TF_ASSERT_OK(Compile(kHloText));

  std::vector<math::bn254::Fr> x = {1, 2, 3, 4, 5, 6};

  std::vector<Literal> literals;
  literals.push_back(LiteralUtil::CreateR1<math::bn254::Fr>(x));
  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, Run(absl::MakeSpan(literals)));

  absl::Span expected = absl::MakeSpan(x).subspan(2, 3);
  EXPECT_EQ(ret_literal, LiteralUtil::CreateR1<math::bn254::Fr>(expected));
}

}  // namespace zkx::cpu
