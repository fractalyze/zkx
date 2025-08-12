/* Copyright 2017 The OpenXLA Authors.

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

#include "zkx/tests/hlo_test_base.h"

#include <utility>

#include "absl/log/check.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/testlib/file_check.h"
#include "zkx/pjrt/pjrt_client.h"
#include "zkx/service/hlo_runner_pjrt.h"
#include "zkx/service/platform_util.h"
#include "zkx/stream_executor/stream_executor_memory_allocator.h"
#include "zkx/tests/pjrt_client_registry.h"

namespace zkx {
namespace {

constexpr std::string_view kInterpreter = "interpreter";

// Returns either an HloRunner or HloRunnerPjRt implementation depending on
// whether there exists a registered PjRtClientFactory.
absl::StatusOr<std::unique_ptr<HloRunnerInterface>> GetHloRunnerForTest(
    se::Platform* test_platform) {
  if (ShouldUsePjRt()) {
    PjRtClientTestFactoryRegistry& pjrt_registry =
        GetGlobalPjRtClientTestFactory();
    TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtClient> client,
                        pjrt_registry.Get()());
    PjRtClientTestFactoryRegistry::DeviceShapeRepresentationFn
        device_shape_representation_fn =
            pjrt_registry.GetDeviceShapeRepresentationFn(client.get());
    PjRtClientTestFactoryRegistry::DeviceShapeSizeFn device_shape_size_fn =
        pjrt_registry.GetDeviceShapeSizeFn(client.get());

    return std::make_unique<HloRunnerPjRt>(std::move(client),
                                           device_shape_representation_fn,
                                           device_shape_size_fn);
  }

  return std::make_unique<HloRunner>(test_platform);
}

absl::StatusOr<std::unique_ptr<HloRunnerInterface>> GetHloRunnerForReference(
    se::Platform* reference_platform) {
  return std::make_unique<HloRunner>(reference_platform);
}

}  // namespace

HloTestBase::HloTestBase(bool verifier_layout_sensitive,
                         HloPredicate instruction_can_change_layout_func)
    : HloTestBase(GetTestPlatform(), GetReferencePlatform(),
                  verifier_layout_sensitive,
                  instruction_can_change_layout_func) {}

HloTestBase::HloTestBase(se::Platform* test_platform,
                         se::Platform* reference_platform,
                         bool verifier_layout_sensitive,
                         HloPredicate instruction_can_change_layout_func)
    : HloRunnerAgnosticReferenceMixin<HloRunnerAgnosticTestBase>(
          /*reference_runner=*/GetHloRunnerForReference(reference_platform)
              .value(),
          /*test_runner=*/GetHloRunnerForTest(test_platform).value(),
          verifier_layout_sensitive),
      test_platform_(test_platform) {}

// static
se::Platform* HloTestBase::GetReferencePlatform() {
  auto result = PlatformUtil::GetPlatform(kInterpreter);
  CHECK_OK(result.status()) << "could not get interpreter platform";
  return result.value();
}

// static
se::Platform* HloTestBase::GetTestPlatform() {
  auto result = PlatformUtil::GetDefaultPlatform();
  CHECK_OK(result.status()) << "could not get test platform";
  return result.value();
}

absl::StatusOr<std::unique_ptr<HloRunnerInterface>>
HloTestBase::GetHloRunner() {
  absl::StatusOr<std::unique_ptr<HloRunnerInterface>> status_or_runner =
      GetHloRunnerForTest(test_platform_);
  // Test for successful creation of PjRt based Hlo Runner.
  CHECK_OK(status_or_runner.status());
  return *std::move(status_or_runner);
}

::testing::AssertionResult HloTestBase::RunAndCompareFromFile(
    const std::string& filename,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto module_or_status =
      HloRunner::ReadModuleFromHloTextFile(filename, GetDebugOptionsForTest());
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "failed reading hlo module from file";
  }
  return RunAndCompare(std::move(module_or_status).value(),
                       reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPassesFromFile(
    const std::string& filename,
    const std::function<void(HloModule*)>& reference_preprocessor) {
  auto module_or_status =
      HloRunner::ReadModuleFromHloTextFile(filename, GetDebugOptionsForTest());
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "failed reading hlo module from file";
  }
  return RunAndCompareNoHloPasses(std::move(module_or_status).value(),
                                  reference_preprocessor);
}

se::DeviceMemoryAllocator* HloTestBase::GetAllocator() {
  if (allocator_ == nullptr) {
    allocator_ = std::make_unique<se::StreamExecutorMemoryAllocator>(
        backend().default_stream_executor());
  }
  return allocator_.get();
}

void HloTestBase::MatchOptimizedHlo(std::string_view hlo,
                                    std::string_view pattern,
                                    bool print_operand_shape) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(hlo));
  HloPrintOptions print_opts;
  print_opts.set_print_operand_shape(print_operand_shape);
  absl::StatusOr<bool> filecheck_result =
      RunFileCheck(optimized_module->ToString(print_opts), pattern);
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.value());
}

absl::StatusOr<std::unique_ptr<HloModule>> HloTestBase::GetOptimizedModule(
    std::string_view hlo) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  // TODO - b/391868033: Remove calls to UpdateEntryComputationLayout.
  UpdateEntryComputationLayout(module.get());
  return backend().compiler()->RunHloPasses(
      std::move(module), backend().default_stream_executor(), GetAllocator());
}

absl::StatusOr<std::unique_ptr<HloModule>> HloTestBase::GetOptimizedModule(
    std::unique_ptr<HloModule> hlo_module) {
  // TODO - b/391868033: Remove calls to UpdateEntryComputationLayout.
  UpdateEntryComputationLayout(hlo_module.get());
  return backend().compiler()->RunHloPasses(std::move(hlo_module),
                                            backend().default_stream_executor(),
                                            GetAllocator());
}

}  // namespace zkx
