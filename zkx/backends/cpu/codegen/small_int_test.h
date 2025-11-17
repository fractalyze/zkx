#ifndef ZKX_BACKENDS_CPU_CODEGEN_SMALL_INT_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_SMALL_INT_TEST_H_

#include "absl/strings/substitute.h"

#include "zkx/backends/cpu/codegen/cpu_kernel_emitter_test.h"
#include "zkx/base/random.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"

namespace zkx::cpu {

template <typename T>
class BaseSmallIntTest {
 protected:
  static T GetRandomValue() { return static_cast<T>(base::Uniform<uint8_t>()); }
};

template <typename T>
class SmallIntScalarBinaryTest : public BaseSmallIntTest<T>,
                                 public CpuKernelEmitterTest {
 public:
  void SetUp() override {
    CpuKernelEmitterTest::SetUp();
    x_typename_ = primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
    x_ = BaseSmallIntTest<T>::GetRandomValue();
    y_ = BaseSmallIntTest<T>::GetRandomValue();
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

 private:
  T x_;
  T y_;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_SMALL_INT_TEST_H_
