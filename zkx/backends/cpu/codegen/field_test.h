#ifndef ZKX_BACKENDS_CPU_CODEGEN_FIELD_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_FIELD_TEST_H_

#include <algorithm>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/substitute.h"

#include "xla/tsl/platform/status.h"
#include "zkx/array2d.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/comparison_util.h"
#include "zkx/literal_util.h"
#include "zkx/math/base/batch_inverse.h"
#include "zkx/math/base/sparse_matrix.h"
#include "zkx/math/poly/root_of_unity.h"
#include "zkx/primitive_util.h"

namespace zkx::cpu {

template <typename F>
class FieldScalarUnaryTest : public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = F::Random();
    literals_.push_back(LiteralUtil::CreateR0<F>(x_));
  }

 protected:
  void SetUpConvert() {
    using FStd = typename F::StdType;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0_std[] convert(%x)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<FStd>(x_.MontReduce());
  }

  void SetUpNegate() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] negate(%x)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(-x_);
  }

  void SetUpInverse() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] inverse(%x)
      }
    )",
                                 x_typename_);
    if (x_.IsZero()) {
      expected_literal_ = LiteralUtil::CreateR0<F>(0);
    } else {
      expected_literal_ = LiteralUtil::CreateR0<F>(*x_.Inverse());
    }
  }

 private:
  F x_;
};

template <typename F>
class FieldScalarBinaryTest : public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = F::Random();
    y_ = F::Random();
    literals_.push_back(LiteralUtil::CreateR0<F>(x_));
    literals_.push_back(LiteralUtil::CreateR0<F>(y_));
  }

 protected:
  void SetUpAdd() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] add(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_ + y_);
  }

  void SetUpCompare() {
    ComparisonDirection direction = RandomComparisonDirection();
    std::string direction_str = ComparisonDirectionToString(direction);

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = pred[] compare(%x, %y), direction=$1
      }
    )",
                                 x_typename_, direction_str);

    switch (direction) {
      case ComparisonDirection::kEq:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ == y_);
        break;
      case ComparisonDirection::kNe:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ != y_);
        break;
      case ComparisonDirection::kGe:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ >= y_);
        break;
      case ComparisonDirection::kGt:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ > y_);
        break;
      case ComparisonDirection::kLe:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ <= y_);
        break;
      case ComparisonDirection::kLt:
        expected_literal_ = LiteralUtil::CreateR0<bool>(x_ < y_);
        break;
    }
  }

  void SetUpDiv() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] divide(%x, %y)
      }
    )",
                                 x_typename_);
    if (y_.IsZero()) {
      expected_literal_ = LiteralUtil::CreateR0<F>(0);
    } else {
      expected_literal_ = LiteralUtil::CreateR0<F>(*(x_ / y_));
    }
  }

  void SetUpMaximum() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] maximum(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(std::max(x_, y_));
  }

  void SetUpMinimum() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] minimum(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(std::min(x_, y_));
  }

  void SetUpMul() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] multiply(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_ * y_);
  }

  void SetUpPow() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = u32[] parameter(1)

        ROOT %ret = $0[] power(%x, %y)
      }
    )",
                                 x_typename_);

    auto y = base::Uniform<uint32_t>();
    literals_[1] = LiteralUtil::CreateR0<uint32_t>(y);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_.Pow(y));
  }

  void SetUpPowWithSignedExponentShouldFail() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = s32[] parameter(1)

        ROOT %ret = $0[] power(%x, %y)
      }
    )",
                                 x_typename_);
    expected_status_code_ = absl::StatusCode::kInvalidArgument;
  }

  void SetUpSub() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] subtract(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_ - y_);
  }

 private:
  F x_;
  F y_;
};

template <typename F>
class FieldScalarTernaryTest : public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = F::Random();
    y_ = F::Random();
    z_ = F::Random();
    literals_.push_back(LiteralUtil::CreateR0<F>(x_));
    literals_.push_back(LiteralUtil::CreateR0<F>(y_));
    literals_.push_back(LiteralUtil::CreateR0<F>(z_));
  }

 protected:
  void SetUpClamp() {
    if (x_ > z_) {
      std::swap(x_, z_);
      std::swap(literals_[0], literals_[2]);
    }
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %min = $0[] parameter(0)
        %operand = $0[] parameter(1)
        %max = $0[] parameter(2)

        ROOT %ret = $0[] clamp(%min, %operand, %max)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(std::clamp(y_, x_, z_));
  }

 private:
  F x_;
  F y_;
  F z_;
};

template <typename F>
class FieldR1TensorUnaryTest : public CpuKernelEmitterTest {
 public:
  constexpr static int64_t N = 4;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = base::CreateVector(N, []() { return F::Random(); });
    literals_.push_back(LiteralUtil::CreateR1<F>(x_));
  }

 protected:
  void Verify(const Literal& ret_literal) const override {
    for (int64_t i = 0; i < N; ++i) {
      if (expected_[i].has_value()) {
        EXPECT_EQ(ret_literal.Get<F>({i}), expected_[i].value());
      }
    }
  }

  void SetUpBatchInverse() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$1] inverse(%x)
      }
    )",
                                 x_typename_, N);

    std::vector<F> expected;
    TF_ASSERT_OK(math::BatchInverse(x_, &expected));

    expected_ = base::Map(expected, [](const absl::StatusOr<F>& x) {
      return std::optional<F>(x.value());
    });
  }

  void SetUpFFT() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$1] fft(%x), fft_type=FFT, fft_length=$1
      }
    )",
                                 x_typename_, N);
    expected_ = base::Map(ComputeFFT(x_),
                          [](const F& x) { return std::optional<F>(x); });
  }

  void SetUpFFTWithTwiddles() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)
        %y = $0[$1] parameter(1)

        ROOT %ret = $0[$1] fft(%x, %y), fft_type=FFT, fft_length=$1
      }
    )",
                                 x_typename_, N);
    std::vector<F> twiddles = ComputeTwiddles(x_);
    literals_.push_back(LiteralUtil::CreateR1<F>(twiddles));
    expected_ = base::Map(ComputeFFT(x_, twiddles),
                          [](const F& x) { return std::optional<F>(x); });
  }

  void SetUpIFFT() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$1] fft(%x), fft_type=IFFT, fft_length=$1
      }
    )",
                                 x_typename_, N);
    expected_ = base::Map(ComputeInverseFFT(x_),
                          [](const F& x) { return std::optional<F>(x); });
  }

  void SetUpIFFTWithTwiddles() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)
        %y = $0[$1] parameter(1)

        ROOT %ret = $0[$1] fft(%x, %y), fft_type=IFFT, fft_length=$1
      }
    )",
                                 x_typename_, N);
    std::vector<F> twiddles = ComputeInverseTwiddles(x_);
    literals_.push_back(LiteralUtil::CreateR1<F>(twiddles));
    expected_ = base::Map(ComputeInverseFFT(x_, twiddles),
                          [](const F& x) { return std::optional<F>(x); });
  }

 private:
  std::vector<F> ComputeTwiddles(const std::vector<F>& x) {
    absl::StatusOr<F> w_or = math::GetRootOfUnity<F>(x.size());
    CHECK_OK(w_or);
    F w = w_or.value();
    std::vector<F> twiddles(x.size());
    F twiddle = 1;
    for (int64_t i = 0; i < x.size(); ++i) {
      twiddles[i] = twiddle;
      twiddle = twiddle * w;
    }
    return twiddles;
  }

  std::vector<F> ComputeFFT(const std::vector<F>& x,
                            const std::vector<F>& twiddles) {
    std::vector<F> ret(x.size());
    for (int64_t i = 0; i < x.size(); ++i) {
      F v = 0;
      for (int64_t j = 0; j < x.size(); ++j) {
        v += x[j] * twiddles[(i * j) % x.size()];
      }
      ret[i] = v;
    }
    return ret;
  }

  std::vector<F> ComputeFFT(const std::vector<F>& x) {
    return ComputeFFT(x, ComputeTwiddles(x));
  }

  std::vector<F> ComputeInverseTwiddles(const std::vector<F>& x) {
    std::vector<F> twiddles = ComputeTwiddles(x);
    CHECK_OK(math::BatchInverse(twiddles, &twiddles));
    return twiddles;
  }

  std::vector<F> ComputeInverseFFT(const std::vector<F>& x,
                                   const std::vector<F>& twiddles) {
    F n_inv = *F(x.size()).Inverse();
    std::vector<F> ret(x.size());
    for (int64_t i = 0; i < x.size(); ++i) {
      F v = 0;
      for (int64_t j = 0; j < x.size(); ++j) {
        v += x[j] * twiddles[(i * j) % x.size()];
      }
      ret[i] = v * n_inv;
    }
    return ret;
  }

  std::vector<F> ComputeInverseFFT(const std::vector<F>& x) {
    return ComputeInverseFFT(x, ComputeInverseTwiddles(x));
  }

  std::vector<F> x_;
  // TODO(chokobole): ZKIR BatchInverse returns garbage value when the input
  // value is zero, which behaves differently from single inverse. If the
  // batch inverse also returns zero output, we should compare the result with
  // Literal.
  std::vector<std::optional<F>> expected_;
};

template <typename F>
class FieldR2TensorBinaryTest : public CpuKernelEmitterTest {
 public:
  constexpr static int64_t M = 2;
  constexpr static int64_t N = 3;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = base::CreateVector(M, []() {
      return base::CreateVector(N, []() { return F::Random(); });
    });
    y_ = base::CreateVector(M, []() {
      return base::CreateVector(N, []() { return F::Random(); });
    });
    Array2D<F> x_array(M, N);
    Array2D<F> y_array(M, N);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        x_array({i, j}) = x_[i][j];
        y_array({i, j}) = y_[i][j];
      }
    }
    literals_.push_back(LiteralUtil::CreateR2FromArray2D(x_array));
    literals_.push_back(LiteralUtil::CreateR2FromArray2D(y_array));
  }

 protected:
  void Verify(const Literal& ret_literal) const override {
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        EXPECT_EQ(ret_literal.Get<F>({i, j}), expected_[i][j]);
      }
    }
  }

  void SetUpAdd() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2] parameter(0)
        %y = $0[$1, $2] parameter(1)

        ROOT %ret = $0[$1, $2] add(%x, %y)
      }
    )",
                                 x_typename_, M, N);

    expected_ = base::CreateVector(M, [this](size_t i) {
      return base::CreateVector(
          N, [this, i](size_t j) { return x_[i][j] + y_[i][j]; });
    });
  }

 private:
  std::vector<std::vector<F>> x_;
  std::vector<std::vector<F>> y_;
  std::vector<std::vector<F>> expected_;
};

template <typename F>
class FieldTest : public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
  }

 protected:
  void SetUpBroadcastScalar() {
    constexpr static int64_t N = 4;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[$1] broadcast(%x)
      }
    )",
                                 x_typename_, N);

    auto x = F::Random();
    literals_.push_back(LiteralUtil::CreateR0<F>(x));
    expected_literal_ = LiteralUtil::CreateR1<F>(
        base::CreateVector(N, [x](size_t i) { return x; }));
  }

  void SetUpBroadcastTensorR1ToR3WithD0() {
    SetUpBroadcastTensorR1ToR3Helper(2, 0, [](const std::vector<F>& x) {
      return LiteralUtil::CreateR3<F>(
          {{{x[0], x[0]}, {x[0], x[0]}}, {{x[1], x[1]}, {x[1], x[1]}}});
    });
  }

  void SetUpBroadcastTensorR1ToR3WithD1() {
    SetUpBroadcastTensorR1ToR3Helper(2, 1, [](const std::vector<F>& x) {
      return LiteralUtil::CreateR3<F>(
          {{{x[0], x[0]}, {x[1], x[1]}}, {{x[0], x[0]}, {x[1], x[1]}}});
    });
  }

  void SetUpBroadcastTensorR1ToR3WithD2() {
    SetUpBroadcastTensorR1ToR3Helper(2, 2, [](const std::vector<F>& x) {
      return LiteralUtil::CreateR3<F>(
          {{{x[0], x[1]}, {x[0], x[1]}}, {{x[0], x[1]}, {x[0], x[1]}}});
    });
  }

  void SetUpCSRMatrixVectorMultiplication() {
    constexpr static int64_t M = 4;
    constexpr static int64_t N = 3;
    constexpr static int64_t NNZ = 8;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1, $2]{1,0:D(D, C) NNZ($3)} parameter(0)
        %y = $0[$2] parameter(1)

        ROOT %ret = $0[$1] dot(%x, %y)
      }
    )",
                                 x_typename_, M, N, NNZ);

    math::SparseMatrix<F> x = math::SparseMatrix<F>::Random(M, N, NNZ);

    std::vector<uint32_t> x_row_ptrs, x_col_indices;
    std::vector<F> x_values;
    x.ToCSR(x_row_ptrs, x_col_indices, x_values);

    TF_ASSERT_OK_AND_ASSIGN(std::vector<uint8_t> x_buffer, x.ToCSRBuffer());
    std::vector<F> y = base::CreateVector(N, []() { return F::Random(); });

    literals_.push_back(LiteralUtil::CreateR1<uint8_t>(x_buffer));
    literals_.push_back(LiteralUtil::CreateR1<F>(y));

    std::vector<F> expected(x_row_ptrs.size() - 1);
    for (int64_t i = 0; i < expected.size(); ++i) {
      for (int64_t j = x_row_ptrs[i]; j < x_row_ptrs[i + 1]; ++j) {
        expected[i] += x_values[j] * y[x_col_indices[j]];
      }
    }
    expected_literal_ = LiteralUtil::CreateR1<F>(expected);
  }

  void SetUpSlice() {
    constexpr static int64_t N = 6;
    constexpr static int64_t S = 2;
    constexpr static int64_t E = 5;

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$2] slice(%x), slice={[$3:$4]}
      }
    )",
                                 x_typename_, N, E - S, S, E);

    auto x = base::CreateVector(N, []() { return F::Random(); });
    literals_.push_back(LiteralUtil::CreateR1<F>(x));
    expected_literal_ = LiteralUtil::CreateR1<F>(
        base::CreateVector(E - S, [&x](size_t i) { return x[i + S]; }));
  }

 private:
  void SetUpBroadcastTensorR1ToR3Helper(
      int64_t m, int64_t d,
      std::function<Literal(const std::vector<F>&)> callback) {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$1, $1, $1] broadcast(%x), dimensions={$2}
      }
    )",
                                 x_typename_, m, d);

    auto x = base::CreateVector(m, []() { return F::Random(); });
    literals_.push_back(LiteralUtil::CreateR1<F>(x));
    expected_literal_ = callback(x);
  }
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_FIELD_TEST_H_
