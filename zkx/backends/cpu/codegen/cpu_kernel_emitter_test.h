#ifndef ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_
#define ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_

#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

#include "zkx/literal.h"
#include "zkx/service/hlo_runner.h"

namespace zkx::cpu {

class CpuKernelEmitterTest : public testing::Test {
 public:
  CpuKernelEmitterTest();

  void RunAndVerify();

 protected:
  virtual void Verify(const Literal& ret_literal) const;

  HloRunner runner_;
  std::string_view x_typename_;
  std::vector<Literal> literals_;
  std::string hlo_text_;
  Literal expected_literal_;
  absl::StatusCode expected_status_code_ = absl::StatusCode::kOk;
};

}  // namespace zkx::cpu

#endif  // ZKX_BACKENDS_CPU_CODEGEN_CPU_KERNEL_EMITTER_TEST_H_
