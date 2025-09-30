#ifndef ZKX_BACKENDS_CPU_CODEGEN_GROUP_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_GROUP_TEST_H_

#include <string_view>

#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::cpu {

template <typename AffinePoint>
class GroupScalarUnaryTest : public CpuKernelEmitterTest {
 public:
  using JacobianPoint = typename AffinePoint::JacobianPoint;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
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
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] convert(%x)
      }
    )",
                                 x_typename_, ret_typename_);
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_.ToJacobian());
  }

  void SetUpNegate() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $0[] negate(%x)
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
class GroupScalarBinaryTest : public CpuKernelEmitterTest {
 public:
  using JacobianPoint = typename AffinePoint::JacobianPoint;
  using ScalarField = typename AffinePoint::ScalarField;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
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
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $1[] add(%x, %y)
      }
    )",
                                 x_typename_, ret_typename_);
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_ + y_);
  }

  void SetUpDouble() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)

        ROOT %ret = $1[] add(%x, %x)
      }
    )",
                                 x_typename_, ret_typename_);
    literals_.pop_back();
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_ + x_);
  }

  void SetUpSub() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $0[] parameter(1)

        ROOT %ret = $1[] subtract(%x, %y)
      }
    )",
                                 x_typename_, ret_typename_);
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(x_ - y_);
  }

  void SetUpScalarMul() {
    hlo_text_ = absl::Substitute(
        R"(
      ENTRY %main {
        %x = $0[] parameter(0)
        %y = $1[] parameter(1)

        ROOT %ret = $2[] multiply(%x, %y)
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

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_GROUP_TEST_H_
