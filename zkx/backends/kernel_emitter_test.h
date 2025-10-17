#ifndef ZKX_BACKENDS_KERNEL_EMITTER_TEST_H_
#define ZKX_BACKENDS_KERNEL_EMITTER_TEST_H_

#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

#include "zkx/literal.h"
#include "zkx/service/hlo_runner.h"

namespace zkx {

class KernelEmitterTest : public testing::Test {
 public:
  explicit KernelEmitterTest(std::string_view platform_name);

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

}  // namespace zkx

#endif  // ZKX_BACKENDS_KERNEL_EMITTER_TEST_H_
