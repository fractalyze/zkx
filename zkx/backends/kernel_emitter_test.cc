#include "zkx/backends/kernel_emitter_test.h"

#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/service/platform_util.h"

namespace zkx {

KernelEmitterTest::KernelEmitterTest(std::string_view platform_name)
    : runner_(PlatformUtil::GetPlatform(platform_name).value()) {}

void KernelEmitterTest::RunAndVerify() {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_text_));

  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> opaque_executable =
      runner_.CreateExecutable(std::move(module),
                               /*run_hlo_passes=*/false);
  if (expected_status_code_ != absl::StatusCode::kOk) {
    EXPECT_THAT(opaque_executable,
                ::absl_testing::StatusIs(expected_status_code_));
    return;
  }

  TF_ASSERT_OK_AND_ASSIGN(Literal ret_literal, runner_.ExecuteWithExecutable(
                                                   opaque_executable->get(),
                                                   absl::MakeSpan(literals_),
                                                   /*profile=*/nullptr));
  Verify(ret_literal);
}

void KernelEmitterTest::Verify(const Literal& ret_literal) const {
  EXPECT_EQ(ret_literal, expected_literal_);
}

}  // namespace zkx
