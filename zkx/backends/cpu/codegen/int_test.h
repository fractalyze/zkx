#ifndef ZKX_BACKENDS_CPU_CODEGEN_INT_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_INT_TEST_H_

#include <type_traits>
#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/substitute.h"

#include "zkx/array2d.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/base/random.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::cpu {

template <typename T>
class BaseIntTest {
 protected:
  using UnsignedT =
      std::conditional_t<std::is_signed_v<T>, std::make_unsigned_t<T>, T>;

  static T GetRandomValue() {
    return absl::bit_cast<T>(base::Uniform<UnsignedT>());
  }
};

template <typename T>
class IntScalarBinaryTest : public BaseIntTest<T>, public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
    x_ = BaseIntTest<T>::GetRandomValue();
    y_ = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<T>(x_));
    literals_.push_back(LiteralUtil::CreateR0<T>(y_));
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
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ + y_);
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
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ - y_);
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
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ * y_);
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

    // Re-sample divisor `y_` until it is safe to divide:
    //  - Disallow zero to avoid division-by-zero.
    //  - For signed T, also disallow the INT_MIN / -1 pattern:
    //      If x_ == std::numeric_limits<T>::min() and y_ == -1,
    //      then x_ / y_ triggers undefined behavior on two's-complement
    //      because -min(T) is not representable.
    while (y_ == 0 || (std::is_signed_v<T> && y_ == -1 &&
                       x_ == std::numeric_limits<T>::min())) {
      y_ = BaseIntTest<T>::GetRandomValue();
      literals_[1] = LiteralUtil::CreateR0<T>(y_);
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ / y_);
  }

 private:
  T x_;
  T y_;
};

template <typename T>
class IntR2TensorBinaryTest : public BaseIntTest<T>,
                              public CpuKernelEmitterTest {
 public:
  constexpr static int64_t M = 2;
  constexpr static int64_t N = 3;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
    x_ = base::CreateVector(M, []() {
      return base::CreateVector(
          N, []() { return BaseIntTest<T>::GetRandomValue(); });
    });
    y_ = base::CreateVector(M, []() {
      return base::CreateVector(
          N, []() { return BaseIntTest<T>::GetRandomValue(); });
    });
    Array2D<T> x_array(M, N);
    Array2D<T> y_array(M, N);
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
        EXPECT_EQ(ret_literal.Get<T>({i, j}), expected_[i][j]);
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
  std::vector<std::vector<T>> x_;
  std::vector<std::vector<T>> y_;
  std::vector<std::vector<T>> expected_;
};

template <typename T>
class IntTest : public BaseIntTest<T>, public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
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

    auto x = base::CreateVector(
        N, []() { return BaseIntTest<T>::GetRandomValue(); });
    literals_.push_back(LiteralUtil::CreateR1<T>(x));
    expected_literal_ = LiteralUtil::CreateR1<T>(
        base::CreateVector(E - S, [&x](size_t i) { return x[i + S]; }));
  }
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_INT_TEST_H_
