#ifndef ZKX_BACKENDS_CPU_CODEGEN_INT_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_INT_TEST_H_

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/substitute.h"

#include "zkx/array2d.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/base/random.h"
#include "zkx/comparison_util.h"
#include "zkx/literal_util.h"
#include "zkx/math/base/pow.h"
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
class IntScalarUnaryTest : public BaseIntTest<T>, public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
    x_ = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<T>(x_));
  }

 protected:
  void SetUpAbs() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] abs(%x)
      }
    )",
                                 x_typename_);
    if constexpr (std::is_signed_v<T>) {
      expected_literal_ = LiteralUtil::CreateR0<T>(std::abs<T>(x_));
    } else {
      LOG(FATAL) << "abs is not supported for unsigned type";
    }
  }

  void SetUpBitcastConvert() {
    using DstType = typename std::conditional_t<
        std::is_signed_v<T>, std::make_unsigned_t<T>, std::make_signed_t<T>>;
    std::string_view dst_typename = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<DstType>());

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] bitcast-convert(%x)
      }
    )",
                                 x_typename_, dst_typename);
    expected_literal_ =
        LiteralUtil::CreateR0<DstType>(absl::bit_cast<DstType>(x_));
  }

  void SetUpCountLeadingZeros() {
    uint32_t case_num = base::Uniform<uint32_t>() % 2;
    if (case_num == 0) {
      x_ = 0;
      literals_[0] = LiteralUtil::CreateR0<T>(x_);
    }
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] count-leading-zeros(%x)
      }
    )",
                                 x_typename_);
    if (x_ == 0) {
      // NOTE(chokobole): __builtin_clz is undefined behavior for 0.
      // See https://gcc.gnu.org/onlinedocs/gcc/Bit-Operation-Builtins.html
      expected_literal_ = LiteralUtil::CreateR0<T>(sizeof(T) * 8);
    } else {
      expected_literal_ = LiteralUtil::CreateR0<T>(__builtin_clz(x_));
    }
  }

  void SetUpConvertUp() {
    using DstType =
        typename std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
    static_assert(sizeof(T) < sizeof(DstType),
                  "T must be smaller than DstType");
    std::string_view dst_typename = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<DstType>());

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] convert(%x)
      }
    )",
                                 x_typename_, dst_typename);

    expected_literal_ =
        LiteralUtil::CreateR0<DstType>(static_cast<DstType>(x_));
  }

  void SetUpConvertDown() {
    using DstType =
        typename std::conditional_t<std::is_signed_v<T>, int16_t, uint16_t>;
    static_assert(sizeof(T) > sizeof(DstType), "T must be larger than DstType");
    std::string_view dst_typename = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<DstType>());

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] convert(%x)
      }
    )",
                                 x_typename_, dst_typename);
    expected_literal_ =
        LiteralUtil::CreateR0<DstType>(static_cast<DstType>(x_));
  }

  void SetUpNegate() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] negate(%x)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(-x_);
  }

  void SetUpNot() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] not(%x)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(~x_);
  }

  void SetUpPopulationCount() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] popcnt(%x)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(__builtin_popcount(x_));
  }

  void SetUpSign() {
    uint32_t case_num = base::Uniform<uint32_t>() % 2;
    if (case_num == 0) {
      x_ = 0;
      literals_[0] = LiteralUtil::CreateR0<T>(x_);
    }
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] sign(%x)
      }
    )",
                                 x_typename_);
    T sign;
    if (x_ == 0) {
      sign = 0;
    } else if (x_ > 0) {
      sign = 1;
    } else {
      sign = -1;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(sign);
  }

 private:
  T x_;
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

  void SetUpAnd() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] and(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ & y_);
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
    uint32_t case_num = base::Uniform<uint32_t>() % 3;
    if (case_num == 0) {
      y_ = 0;
      literals_[1] = LiteralUtil::CreateR0<T>(y_);
    } else if (case_num == 1) {
      if (std::is_signed_v<T>) {
        x_ = std::numeric_limits<T>::min();
        literals_[0] = LiteralUtil::CreateR0<T>(x_);
        y_ = -1;
        literals_[1] = LiteralUtil::CreateR0<T>(y_);
      }
    }

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] divide(%x, %y)
      }
    )",
                                 x_typename_);

    T expected;
    if (y_ == 0) {
      expected = -1;
    } else if (std::is_signed_v<T> && x_ == std::numeric_limits<T>::min() &&
               y_ == -1) {
      expected = std::numeric_limits<T>::min();
    } else {
      expected = x_ / y_;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
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
    expected_literal_ = LiteralUtil::CreateR0<T>(std::max(x_, y_));
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
    expected_literal_ = LiteralUtil::CreateR0<T>(std::min(x_, y_));
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

  void SetUpOr() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] or(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ | y_);
  }

  void SetUpPower() {
    uint32_t case_num = base::Uniform<uint32_t>() % 3;
    if (case_num == 0) {
      x_ = -1;
      literals_[0] = LiteralUtil::CreateR0<T>(x_);
    } else if (case_num == 1) {
      x_ = 1;
      literals_[0] = LiteralUtil::CreateR0<T>(x_);
    }
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] power(%x, %y)
      }
    )",
                                 x_typename_);
    T expected;
    if (x_ == -1) {
      expected = y_ % 2 == 0 ? 1 : -1;
    } else if (x_ == 1) {
      expected = 1;
    } else if (y_ >= 0) {
      expected = math::Pow(x_, y_);
    } else {
      expected = 0;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
  }

  void SetUpRemainder() {
    uint32_t case_num = base::Uniform<uint32_t>() % 3;
    if (case_num == 0) {
      y_ = 0;
      literals_[1] = LiteralUtil::CreateR0<T>(y_);
    } else if (case_num == 1) {
      if (std::is_signed_v<T>) {
        x_ = std::numeric_limits<T>::min();
        literals_[0] = LiteralUtil::CreateR0<T>(x_);
        y_ = -1;
        literals_[1] = LiteralUtil::CreateR0<T>(y_);
      }
    }

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] remainder(%x, %y)
      }
    )",
                                 x_typename_);

    T expected;
    if (y_ == 0) {
      expected = x_;
    } else if (std::is_signed_v<T> && x_ == std::numeric_limits<T>::min() &&
               y_ == -1) {
      expected = 0;
    } else {
      expected = x_ % y_;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
  }

  void SetUpShiftLeft() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] shift-left(%x, %y)
      }
    )",
                                 x_typename_);

    T expected;
    if (IsShiftOutOfBounds(y_)) {
      expected = 0;
    } else {
      expected = x_ << y_;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
  }

  void SetUpShiftRightArithmetic() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] shift-right-arithmetic(%x, %y)
      }
    )",
                                 x_typename_);

    T expected;
    if (IsShiftOutOfBounds(y_)) {
      using SignedT = std::make_signed_t<T>;
      auto x_signed = static_cast<SignedT>(x_);
      if (x_signed < 0) {
        expected = -1;
      } else {
        expected = 0;
      }
    } else {
      expected = x_ >> y_;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
  }

  void SetUpShiftRightLogical() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] shift-right-logical(%x, %y)
      }
    )",
                                 x_typename_);

    T expected;
    if (IsShiftOutOfBounds(y_)) {
      expected = 0;
    } else {
      expected = x_ >> y_;
    }
    expected_literal_ = LiteralUtil::CreateR0<T>(expected);
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

  void SetUpXor() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $0[] xor(%x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(x_ ^ y_);
  }

 private:
  static bool IsShiftOutOfBounds(T rhs) {
    using UnsignedT = std::make_unsigned_t<T>;
    UnsignedT lhs_bits_unsigned =
        static_cast<UnsignedT>(std::numeric_limits<UnsignedT>::digits);
    UnsignedT rhs_unsigned = static_cast<UnsignedT>(rhs);
    return rhs_unsigned >= lhs_bits_unsigned;
  }

  T x_;
  T y_;
};

template <typename T>
class IntScalarTernaryTest : public BaseIntTest<T>,
                             public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
    x_ = BaseIntTest<T>::GetRandomValue();
    y_ = BaseIntTest<T>::GetRandomValue();
    z_ = BaseIntTest<T>::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<T>(x_));
    literals_.push_back(LiteralUtil::CreateR0<T>(y_));
    literals_.push_back(LiteralUtil::CreateR0<T>(z_));
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
    expected_literal_ = LiteralUtil::CreateR0<T>(std::clamp(y_, x_, z_));
  }

  void SetUpSelect() {
    bool cond = x_ % 2 == 0;
    literals_[0] = LiteralUtil::CreateR0<bool>(cond);
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %cond = pred[] parameter(0)
        %x = $0[] parameter(1)
        %y = $0[] parameter(2)

        ROOT %ret = $0[] select(%cond, %x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<T>(cond ? y_ : z_);
  }

 private:
  T x_;
  T y_;
  T z_;
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

  void SetUpConditional() {
    hlo_text_ = absl::Substitute(R"(
      %identity {
        ROOT %ret = $0[] parameter(0)
      }

      %negate {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] negate(%x)
      }

      ENTRY %main {
        %cond = pred[] parameter(0)
        %x = $0[] parameter(1)

        ROOT %ret = $0[] conditional(%cond, %x, %x), true_computation=%identity, false_computation=%negate
      }
    )",
                                 x_typename_);

    auto x = BaseIntTest<T>::GetRandomValue();
    bool cond = base::Uniform<uint32_t>() % 2 == 0;
    literals_.push_back(LiteralUtil::CreateR0<bool>(cond));
    literals_.push_back(LiteralUtil::CreateR0<T>(x));
    expected_literal_ = LiteralUtil::CreateR0<T>(cond ? x : -x);
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

  void SetUpWhile() {
    hlo_text_ = absl::Substitute(R"(
      %condition {
        %param = (u32[], u32[], $0[], $0[]) parameter(0)
        %i = u32[] get-tuple-element(%param), index=0
        %n = u32[] get-tuple-element(%param), index=1

        ROOT ret = pred[] compare(%i, %n), direction=LT
      }

      %body {
        %param = (u32[], u32[], $0[], $0[]) parameter(0)
        %i = u32[] get-tuple-element(%param), index=0
        %n = u32[] get-tuple-element(%param), index=1
        %acc = $0[] get-tuple-element(%param), index=2
        %x = $0[] get-tuple-element(%param), index=3

        %one = u32[] constant(1)
        %next_i = u32[] add(%i, %one)
        %new_acc = $0[] add(%acc, %x)
        ROOT %ret = (u32[], u32[], $0[], $0[]) tuple(%next_i, %n, %new_acc, %x)
      }

      ENTRY %main {
        %zero = u32[] constant(0)
        %init = u32[] constant(0)
        %n = u32[] parameter(0)
        %x = $0[] parameter(1)

        %while.tuple = (u32[], u32[], $0[], $0[]) tuple(%zero, %n, %init, %x)
        %result = (u32[], u32[], $0[], $0[]) while(%while.tuple), condition=%condition, body=%body
        ROOT %ret = $0[] get-tuple-element(%result), index=2
      }
    )",
                                 x_typename_);

    auto x = BaseIntTest<T>::GetRandomValue();
    auto n = base::Uniform<uint32_t>() % 10;
    literals_.push_back(LiteralUtil::CreateR0<uint32_t>(n));
    literals_.push_back(LiteralUtil::CreateR0<T>(x));
    expected_literal_ = LiteralUtil::CreateR0<T>(n * x);
  }
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_INT_TEST_H_
