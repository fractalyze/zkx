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

#ifndef ZKX_BACKENDS_CPU_CODEGEN_EXT_FIELD_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_EXT_FIELD_TEST_H_

#include "absl/strings/substitute.h"

#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::cpu {

template <typename F>
class ExtFieldScalarBinaryTest : public CpuKernelEmitterTest {
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

  void SetUpCompareEq() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = pred[] compare(%x, %y), direction=EQ
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<bool>(x_ == y_);
  }

  void SetUpCompareNe() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = pred[] compare(%x, %y), direction=NE
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<bool>(x_ != y_);
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
      expected_literal_ = LiteralUtil::CreateR0<F>(F::Zero());
    } else {
      expected_literal_ = LiteralUtil::CreateR0<F>(*(x_ / y_));
    }
  }

  void SetUpDouble() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] add(%x, %x)
      }
    )",
                                 x_typename_);
    literals_.pop_back();
    expected_literal_ = LiteralUtil::CreateR0<F>(x_.Double());
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

  void SetUpSquare() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] multiply(%x, %x)
      }
    )",
                                 x_typename_);
    literals_.pop_back();
    expected_literal_ = LiteralUtil::CreateR0<F>(x_.Square());
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
class ExtFieldScalarTernaryTest : public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<F>());
    x_ = F::Random();
    y_ = F::Random();
    literals_.emplace_back();  // Placeholder for the predicate.
    literals_.push_back(LiteralUtil::CreateR0<F>(x_));
    literals_.push_back(LiteralUtil::CreateR0<F>(y_));
  }

 protected:
  void SetUpSelectTrue() {
    literals_[0] = LiteralUtil::CreateR0<bool>(true);
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %cond = pred[] parameter(0)
        %x = $0[] parameter(1)
        %y = $0[] parameter(2)

        ROOT %ret = $0[] select(%cond, %x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(x_);
  }

  void SetUpSelectFalse() {
    literals_[0] = LiteralUtil::CreateR0<bool>(false);
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %cond = pred[] parameter(0)
        %x = $0[] parameter(1)
        %y = $0[] parameter(2)

        ROOT %ret = $0[] select(%cond, %x, %y)
      }
    )",
                                 x_typename_);
    expected_literal_ = LiteralUtil::CreateR0<F>(y_);
  }

 private:
  F x_;
  F y_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_EXT_FIELD_TEST_H_
