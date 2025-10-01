#ifndef ZKX_BACKENDS_GPU_CODEGEN_GROUP_TEST_H_
#define ZKX_BACKENDS_GPU_CODEGEN_GROUP_TEST_H_

#include <string_view>

#include "absl/strings/substitute.h"

#include "zkx/array2d.h"
#include "zkx/backends/gpu/codegen/cuda_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::gpu {

template <typename AffinePoint>
class GroupScalarUnaryTest : public CudaKernelEmitterTest {
 public:
  using JacobianPoint = typename AffinePoint::JacobianPoint;

  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    ret_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<JacobianPoint>());
    x_ = AffinePoint::Random();
    literals_.push_back(LiteralUtil::CreateR0<AffinePoint>(x_));
  }

 protected:
  void SetUpConvert() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] convert(%x)
      }

      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] fusion(%x), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_, ret_typename_);
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_.ToJacobian());
  }

  void SetUpNegate() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] negate(%x)
      }

      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] fusion(%x), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<AffinePoint>(-x_);
  }

 private:
  std::string_view ret_typename_;
  AffinePoint x_;
};

template <typename AffinePoint>
class GroupScalarBinaryTest : public CudaKernelEmitterTest {
 public:
  using JacobianPoint = typename AffinePoint::JacobianPoint;
  using ScalarField = typename AffinePoint::ScalarField;

  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    ret_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<JacobianPoint>());
    x_ = AffinePoint::Random();
    y_ = AffinePoint::Random();
    literals_.push_back(LiteralUtil::CreateR0<AffinePoint>(x_));
    literals_.push_back(LiteralUtil::CreateR0<AffinePoint>(y_));
  }

 protected:
  void SetUpAdd() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $1[] add(%x, %y)
      }

      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $1[] fusion(%x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_, ret_typename_);
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_ + y_);
  }

  void SetUpDouble() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] add(%x, %x)
      }

      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] fusion(%x), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_, ret_typename_);
    literals_.pop_back();
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_ + x_);
  }

  void SetUpSub() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $1[] subtract(%x, %y)
      }

      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $1[] fusion(%x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_, ret_typename_);
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_ - y_);
  }

  void SetUpScalarMul() {
    hlo_text_ = absl::Substitute(
        R"(
      %f {
        %x = $0[] parameter(0)
        %y = $1[] parameter(1)

        ROOT %ret = $2[] multiply(%x, %y)
      }

      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $1[] parameter(1)

        ROOT %ret = $2[] fusion(%x, %y), kind=kLoop, calls=%f
      }
    )",
        primitive_util::LowercasePrimitiveTypeName(
            primitive_util::NativeToPrimitiveType<ScalarField>()),
        primitive_util::LowercasePrimitiveTypeName(
            primitive_util::NativeToPrimitiveType<AffinePoint>()),
        ret_typename_);
    auto x = ScalarField::Random();
    literals_[0] = LiteralUtil::CreateR0<ScalarField>(x);
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x * y_);
  }

 private:
  std::string_view ret_typename_;
  AffinePoint x_;
  AffinePoint y_;
};

template <typename AffinePoint>
class GroupR2TensorUnaryTest : public CudaKernelEmitterTest {
 public:
  constexpr static int64_t M = 2;
  constexpr static int64_t N = 3;

  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    x_ = base::CreateVector(M, []() {
      return base::CreateVector(N, []() { return AffinePoint::Random(); });
    });
    Array2D<AffinePoint> x_array(M, N);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        x_array({i, j}) = x_[i][j];
      }
    }
    literals_.push_back(LiteralUtil::CreateR2FromArray2D(x_array));
  }

 protected:
  void Verify(const Literal& ret_literal) const override {
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        EXPECT_EQ(ret_literal.Get<AffinePoint>({i, j}), expected_[i][j]);
      }
    }
  }

  void SetUpNegate() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[$1, $2] parameter(0)

        ROOT %ret = $0[$1, $2] negate(%x)
      }

      ENTRY %main {
        %x = $0[$1, $2] parameter(0)

        ROOT %ret = $0[$1, $2] fusion(%x), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_, M, N);
    expected_ = base::CreateVector(M, [this](size_t i) {
      return base::CreateVector(N, [this, i](size_t j) { return -x_[i][j]; });
    });
  }

 private:
  std::vector<std::vector<AffinePoint>> x_;
  std::vector<std::vector<AffinePoint>> expected_;
};

template <typename AffinePoint>
class GroupR2TensorBinaryTest : public CudaKernelEmitterTest {
 public:
  constexpr static int64_t M = 2;
  constexpr static int64_t N = 3;

  using JacobianPoint = typename AffinePoint::JacobianPoint;

  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    ret_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<JacobianPoint>());
    x_ = base::CreateVector(M, []() {
      return base::CreateVector(N, []() { return AffinePoint::Random(); });
    });
    y_ = base::CreateVector(M, []() {
      return base::CreateVector(N, []() { return AffinePoint::Random(); });
    });
    Array2D<AffinePoint> x_array(M, N);
    Array2D<AffinePoint> y_array(M, N);
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
        EXPECT_EQ(ret_literal.Get<JacobianPoint>({i, j}), expected_[i][j]);
      }
    }
  }

  void SetUpAdd() {
    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[$2, $3] parameter(0)
        %y = $0[$2, $3] parameter(1)

        ROOT %ret = $1[$2, $3] add(%x, %y)
      }

      ENTRY %main {
        %x = $0[$2, $3] parameter(0)
        %y = $0[$2, $3] parameter(1)

        ROOT %ret = $1[$2, $3] fusion(%x, %y), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_, ret_typename_, M, N);
    expected_ = base::CreateVector(M, [this](size_t i) {
      return base::CreateVector(
          N, [this, i](size_t j) { return x_[i][j] + y_[i][j]; });
    });
  }

 private:
  std::string_view ret_typename_;
  std::vector<std::vector<AffinePoint>> x_;
  std::vector<std::vector<AffinePoint>> y_;
  std::vector<std::vector<JacobianPoint>> expected_;
};

template <typename AffinePoint>
class GroupTest : public CudaKernelEmitterTest {
 public:
  void SetUp() override {
    CudaKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
  }

 protected:
  void SetUpSlice() {
    constexpr static int64_t N = 6;
    constexpr static int64_t S = 2;
    constexpr static int64_t E = 5;

    hlo_text_ = absl::Substitute(R"(
      %f {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$2] slice(%x), slice={[$3:$4]}
      }

      ENTRY %main {
        %x = $0[$1] parameter(0)

        ROOT %ret = $0[$2] fusion(%x), kind=kLoop, calls=%f
      }
    )",
                                 x_typename_, N, E - S, S, E);

    auto x = base::CreateVector(N, []() { return AffinePoint::Random(); });
    literals_.push_back(LiteralUtil::CreateR1<AffinePoint>(x));
    expected_literal_ = LiteralUtil::CreateR1<AffinePoint>(
        base::CreateVector(E - S, [&x](size_t i) { return x[i + S]; }));
  }
};

}  // namespace zkx::gpu

#endif  // ZKX_BACKENDS_GPU_CODEGEN_GROUP_TEST_H_
