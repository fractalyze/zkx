/* Copyright 2017 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_TESTS_HLO_TEST_BASE_H_
#define ZKX_TESTS_HLO_TEST_BASE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/literal.h"
#include "zkx/service/backend.h"
#include "zkx/service/computation_placer.h"
#include "zkx/service/hlo_runner.h"
#include "zkx/service/hlo_runner_interface.h"
#include "zkx/stream_executor/device_memory_allocator.h"
#include "zkx/stream_executor/platform.h"
#include "zkx/tests/hlo_runner_agnostic_reference_mixin.h"
#include "zkx/tests/hlo_runner_agnostic_test_base.h"
#include "zkx/util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

// A base class for tests which build and/or run HLO code. The class includes
// support for running an HLO module on two platforms and comparing the results.
// This is a lower level of abstraction than using the client interface and
// enables, for one, explicitly building a graph of HLO instructions to run.
//
// This can also be used to write text/file-based test cases. Note that the test
// target is responsible for linking the needed backends. A convenient way to do
// this is to make it a zkx_cc_test: it will generate test targets linking with
// the respective backends, which will be used as the test backend; the
// interpreter backend is already linked with hlo_test_base so it will be the
// default reference backend. For example, if you want to compare both cpu vs.
// interpreter, and gpu vs. interpreter, you can:
//
//  zkx_cc_test (
//    name = "sample_text_test",
//    srcs = ["sample_text_test.cc"],
//    backends = [
//      "cpu",
//      "gpu",
//    ],
//    deps = [
//      "//xla/tests:hlo_test_base",
//      ...
//    ],
//  )
//
// For a more detailed example, see "../tests/sample_text_test.cc".
//
// ** NOTE **
// This class will soon be deprecated in favor of HloRunnerAgnosticTestBase. We
// are in the process of incrementally migrating tests to use this new base
// class. HloTestBase remains as a shim on tests during this migration process.
// While we would prefer if you can avoid introducing new tests that use this
// class, we are still working on documenting the exact migration procedure.
class HloTestBase
    : public HloRunnerAgnosticReferenceMixin<HloRunnerAgnosticTestBase> {
 public:
  // Compiles the given `hlo` with optimizations, and verifies that optimized
  // HLO matches the given FileCheck pattern.
  void MatchOptimizedHlo(std::string_view hlo, std::string_view pattern,
                         bool print_operand_shape = false);

  // Like MatchOptimizedHlo, but checks operand shapes as well.
  void MatchOptimizedHloWithShapes(std::string_view hlo,
                                   std::string_view pattern) {
    MatchOptimizedHlo(hlo, pattern, /*print_operand_shape=*/true);
  }

  // Compiles and returns module with optimizations from a given HLO.
  absl::StatusOr<std::unique_ptr<HloModule>> GetOptimizedModule(
      std::string_view hlo);

  absl::StatusOr<std::unique_ptr<HloModule>> GetOptimizedModule(
      std::unique_ptr<HloModule> hlo_module);

  using HloRunnerAgnosticTestBase::ParseAndReturnVerifiedModule;

 protected:
  // This uses the interpreter backend as the reference backend and
  // automatically finds another supported backend as the test backend. If the
  // interpreter is the only supported backend, it will be both the test backend
  // and the reference backend.
  explicit HloTestBase(bool verifier_layout_sensitive = false,
                       HloPredicate instruction_can_change_layout_func = {});

  // If your test doesn't use interpreter as the reference backend, you can use
  // this constructor. Note that your test target is responsible for linking in
  // both needed backends.
  HloTestBase(se::Platform* test_platform, se::Platform* reference_platform,
              bool verifier_layout_sensitive = false,
              HloPredicate instruction_can_change_layout_func = {});

  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[nodiscard]] [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  absl::StatusOr<std::vector<Literal>> ExecuteReplicatedWithHloRunner(
      OpaqueExecutable* executable,
      const HloRunnerInterface::ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment,
      ExecutionProfile* profile = nullptr) {
    return test_runner_as_hlo_runner().ExecuteReplicated(
        executable, options, device_assignment, profile);
  }

  [[nodiscard]] ::testing::AssertionResult RunAndCompareFromFile(
      const std::string& filename,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);
  [[nodiscard]] ::testing::AssertionResult RunAndCompareNoHloPassesFromFile(
      const std::string& filename,
      const std::function<void(HloModule*)>& reference_preprocessor = nullptr);

  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  const Backend& backend() const {
    return test_runner_as_hlo_runner().backend();
  }
  // Returns the backend owned by the test runner.
  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  Backend& backend() {
    return test_runner_as_hlo_runner().backend();
  }

  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  const HloRunner& test_runner_as_hlo_runner() const {
    return *static_cast<HloRunner*>(&test_runner());
  }
  // DO NOT USE: This is a temporary method to help migrate away from HloRunner.
  // Some test fixures rely on functionality that is not supported by other
  // HloRunnerInterface implementations, thus we expose it here.
  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  HloRunner& test_runner_as_hlo_runner() {
    return *static_cast<HloRunner*>(&test_runner());
  }

  [[deprecated(
      "This is a temporary method to help migrate existing tests away from "
      "directly depending on HloRunner. Please do not introduce new uses.")]]
  int64_t num_devices() {
    return backend().device_count();
  }

  absl::StatusOr<std::unique_ptr<HloRunnerInterface>> GetHloRunner();

  // Helper functions to get test and reference platforms.
  static se::Platform* GetReferencePlatform();
  static se::Platform* GetTestPlatform();

  // Creates or retrieves the allocator.
  se::DeviceMemoryAllocator* GetAllocator();

 private:
  se::Platform* test_platform_;  // not owned
  std::unique_ptr<se::DeviceMemoryAllocator> allocator_;
};

}  // namespace zkx

#endif  // ZKX_TESTS_HLO_TEST_BASE_H_
