#ifndef ZKX_BACKENDS_CPU_CODEGEN_MSM_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_MSM_TEST_H_

#include <stddef.h>

#include <string_view>
#include <vector>

#include "xla/tsl/platform/cpu_info.h"
#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/containers/container_util.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::cpu {

template <typename AffinePoint>
class MSMTest : public CpuKernelEmitterTest {
 public:
  using ScalarField = typename AffinePoint::ScalarField;
  using JacobianPoint = typename AffinePoint::JacobianPoint;

  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<ScalarField>());
    y_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<AffinePoint>());
    ret_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<JacobianPoint>());
    num_scalar_muls_ = tsl::port::MaxParallelism() + 1;
    x_ = base::CreateVector(num_scalar_muls_,
                            []() { return ScalarField::Random(); });
    y_ = base::CreateVector(num_scalar_muls_,
                            []() { return AffinePoint::Random(); });
    literals_.push_back(LiteralUtil::CreateR1<ScalarField>(x_));
    literals_.push_back(LiteralUtil::CreateR1<AffinePoint>(y_));
  }

 protected:
  void SetUpMSM() {
    hlo_text_ = absl::Substitute(R"(
      ENTRY %main {
        %x = $0[$3] parameter(0)
        %y = $1[$3] parameter(1)

        ROOT %ret = $2[] msm(%x, %y)
      }
    )",
                                 x_typename_, y_typename_, ret_typename_,
                                 num_scalar_muls_);
    JacobianPoint ret;
    for (size_t i = 0; i < x_.size(); ++i) {
      ret += x_[i] * y_[i];
    }
    expected_literal_ = LiteralUtil::CreateR0<JacobianPoint>(ret);
  }

 private:
  std::string_view y_typename_;
  std::string_view ret_typename_;
  size_t num_scalar_muls_;
  std::vector<ScalarField> x_;
  std::vector<AffinePoint> y_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_MSM_TEST_H_
