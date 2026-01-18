/* Copyright 2026 The ZKX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ZKX_BACKENDS_CPU_CODEGEN_PRED_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_PRED_TEST_H_

#include <vector>

#include "absl/strings/substitute.h"

#include "zkx/array2d.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/base/random.h"
#include "zkx/comparison_util.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::cpu {

class BasePredTest {
 protected:
  static bool GetRandomValue() { return base::Uniform<uint32_t>() % 2 == 0; }
};

class PredScalarUnaryTest : public BasePredTest, public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<bool>());
    x_ = BasePredTest::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<bool>(x_));
  }

 protected:
  void SetUpNot() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = pred[] parameter(0)

        ROOT %ret = pred[] not(%x)
      }
    )");
    expected_literal_ = LiteralUtil::CreateR0<bool>(!x_);
  }

 private:
  bool x_;
};

class PredScalarBinaryTest : public BasePredTest, public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<bool>());
    x_ = BasePredTest::GetRandomValue();
    y_ = BasePredTest::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<bool>(x_));
    literals_.push_back(LiteralUtil::CreateR0<bool>(y_));
  }

 protected:
  void SetUpAnd() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = pred[] parameter(0)
        %y = pred[] parameter(1)

        ROOT %ret = pred[] and(%x, %y)
      }
    )");
    expected_literal_ = LiteralUtil::CreateR0<bool>(x_ && y_);
  }

  void SetUpOr() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = pred[] parameter(0)
        %y = pred[] parameter(1)

        ROOT %ret = pred[] or(%x, %y)
      }
    )");
    expected_literal_ = LiteralUtil::CreateR0<bool>(x_ || y_);
  }

  void SetUpXor() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = pred[] parameter(0)
        %y = pred[] parameter(1)

        ROOT %ret = pred[] xor(%x, %y)
      }
    )");
    expected_literal_ = LiteralUtil::CreateR0<bool>(x_ != y_);
  }

  void SetUpCompare() {
    ComparisonDirection direction = RandomComparisonDirection();
    std::string direction_str = ComparisonDirectionToString(direction);

    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = pred[] parameter(0)
        %y = pred[] parameter(1)

        ROOT %ret = pred[] compare(%x, %y), direction=$0
      }
    )",
                                 direction_str);

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

 private:
  bool x_;
  bool y_;
};

class PredScalarTernaryTest : public BasePredTest, public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<bool>());
    x_ = BasePredTest::GetRandomValue();
    y_ = BasePredTest::GetRandomValue();
    z_ = BasePredTest::GetRandomValue();
    literals_.push_back(LiteralUtil::CreateR0<bool>(x_));
    literals_.push_back(LiteralUtil::CreateR0<bool>(y_));
    literals_.push_back(LiteralUtil::CreateR0<bool>(z_));
  }

 protected:
  void SetUpSelect() {
    bool cond = x_;
    literals_[0] = LiteralUtil::CreateR0<bool>(cond);
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %cond = pred[] parameter(0)
        %x = pred[] parameter(1)
        %y = pred[] parameter(2)

        ROOT %ret = pred[] select(%cond, %x, %y)
      }
    )");
    expected_literal_ = LiteralUtil::CreateR0<bool>(cond ? y_ : z_);
  }

 private:
  bool x_;
  bool y_;
  bool z_;
};

class PredR2TensorBinaryTest : public BasePredTest,
                               public CpuKernelEmitterTest {
 public:
  constexpr static int64_t M = 2;
  constexpr static int64_t N = 3;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<bool>());
    x_ = base::CreateVector(M, []() {
      return base::CreateVector(
          N, []() { return BasePredTest::GetRandomValue(); });
    });
    y_ = base::CreateVector(M, []() {
      return base::CreateVector(
          N, []() { return BasePredTest::GetRandomValue(); });
    });
    Array2D<bool> x_array(M, N);
    Array2D<bool> y_array(M, N);
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
        EXPECT_EQ(ret_literal.Get<bool>({i, j}), expected_[i][j]);
      }
    }
  }

  void SetUpAnd() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = pred[$0, $1] parameter(0)
        %y = pred[$0, $1] parameter(1)

        ROOT %ret = pred[$0, $1] and(%x, %y)
      }
    )",
                                 M, N);

    expected_ = base::CreateVector(M, [this](size_t i) {
      return base::CreateVector(
          N, [this, i](size_t j) { return x_[i][j] && y_[i][j]; });
    });
  }

  void SetUpOr() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = pred[$0, $1] parameter(0)
        %y = pred[$0, $1] parameter(1)

        ROOT %ret = pred[$0, $1] or(%x, %y)
      }
    )",
                                 M, N);

    expected_ = base::CreateVector(M, [this](size_t i) {
      return base::CreateVector(
          N, [this, i](size_t j) { return x_[i][j] || y_[i][j]; });
    });
  }

  void SetUpXor() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = pred[$0, $1] parameter(0)
        %y = pred[$0, $1] parameter(1)

        ROOT %ret = pred[$0, $1] xor(%x, %y)
      }
    )",
                                 M, N);

    expected_ = base::CreateVector(M, [this](size_t i) {
      return base::CreateVector(
          N, [this, i](size_t j) { return x_[i][j] != y_[i][j]; });
    });
  }

 private:
  std::vector<std::vector<bool>> x_;
  std::vector<std::vector<bool>> y_;
  std::vector<std::vector<bool>> expected_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_PRED_TEST_H_
