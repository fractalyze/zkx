/* Copyright 2025 The ZKX Authors.

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

void KernelEmitterTest::RunAndVerify(bool run_hlo_passes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_text_));

  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> opaque_executable =
      runner_.CreateExecutable(std::move(module), run_hlo_passes);
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
