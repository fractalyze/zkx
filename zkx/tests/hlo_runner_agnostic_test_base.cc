/* Copyright 2024 The OpenXLA Authors.
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

#include "zkx/tests/hlo_runner_agnostic_test_base.h"

#include <iterator>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/service/executable.h"
#include "zkx/service/hlo_module_util.h"
#include "zkx/service/hlo_verifier.h"
#include "zkx/tests/literal_test_util.h"
#include "zkx/tests/test_utils.h"

namespace zkx {

HloRunnerAgnosticTestBase::HloRunnerAgnosticTestBase(
    absl_nonnull std::unique_ptr<HloRunnerInterface> test_runner,
    bool verifier_layout_sensitive,
    HloPredicate instruction_can_change_layout_func)
    : HloHardwareIndependentTestBase(verifier_layout_sensitive,
                                     instruction_can_change_layout_func),
      test_runner_(std::move(test_runner)) {}

std::unique_ptr<VerifiedHloModule>
HloRunnerAgnosticTestBase::CreateNewVerifiedModule(const std::string& name,
                                                   int64_t replica_count) {
  return std::make_unique<VerifiedHloModule>(
      name, GetModuleConfigForTest(replica_count), verifier_layout_sensitive(),
      test_runner_->device_shape_size_fn(),
      instruction_can_change_layout_func());
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
HloRunnerAgnosticTestBase::ParseAndReturnVerifiedModule(
    std::string_view hlo_text, int64_t replica_count, int64_t num_partitions) {
  return ParseAndReturnVerifiedModule(
      hlo_text, GetModuleConfigForTest(replica_count, num_partitions));
}

absl::StatusOr<std::unique_ptr<VerifiedHloModule>>
HloRunnerAgnosticTestBase::ParseAndReturnVerifiedModule(
    std::string_view hlo_text, const HloModuleConfig& config,
    const HloParserOptions& parser_options) {
  auto module = std::make_unique<VerifiedHloModule>(
      TestName(), config, verifier_layout_sensitive(),
      test_runner_->device_shape_size_fn(),
      instruction_can_change_layout_func());
  TF_RETURN_IF_ERROR(
      module->ParseHloStringAndVerifyModule(hlo_text, parser_options));
  return std::move(module);
}

HloComputation*
HloRunnerAgnosticTestBase::AddEntryComputationAndUpdateEntryComputationLayout(
    HloModule* module, std::unique_ptr<HloComputation> computation) {
  HloComputation* comp = module->AddEntryComputation(std::move(computation));
  UpdateEntryComputationLayout(module);
  return comp;
}

void HloRunnerAgnosticTestBase::UpdateEntryComputationLayout(
    HloModule* module) const {
  // TODO - b/391868033: Remove UpdateEntryComputationLayout from this class.
  zkx::UpdateEntryComputationLayout(
      module, test_runner_->device_shape_representation_fn());
}

absl::StatusOr<Literal> HloRunnerAgnosticTestBase::Execute(
    std::unique_ptr<HloModule> module,
    absl::Span<const Literal* const> arguments, bool run_hlo_passes) {
  return test_runner_->Execute(std::move(module), arguments, run_hlo_passes);
}

Literal HloRunnerAgnosticTestBase::ExecuteNoHloPasses(
    std::unique_ptr<HloModule> module,
    absl::Span<const Literal* const> arguments) {
  absl::StatusOr<Literal> result = Execute(std::move(module), arguments,
                                           /*run_hlo_passes=*/false);
  CHECK_OK(result.status());
  return *std::move(result);
}

Literal HloRunnerAgnosticTestBase::ExecuteAndTransfer(
    std::unique_ptr<HloModule> module,
    absl::Span<const Literal* const> arguments) {
  absl::StatusOr<Literal> result =
      test_runner_->Execute(std::move(module), arguments, true, nullptr);
  CHECK_OK(result.status());
  return *std::move(result);
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    absl::Span<const Literal* const> arguments, int64_t num_replicas,
    bool use_threads, bool run_hlo_passes) {
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.arguments = {arguments.begin(), arguments.end()};
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = use_threads;
  return test_runner_->ExecuteReplicated(std::move(module), std::move(options));
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    absl::Span<const Literal* const> arguments, int64_t num_replicas,
    DeviceAssignment* device_assignment, bool run_hlo_passes,
    bool use_threads) {
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.arguments = {arguments.begin(), arguments.end()};
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = use_threads;
  return test_runner_->ExecuteReplicated(std::move(module), std::move(options),
                                         device_assignment);
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    std::function<OpaqueExecutable*(int64_t)> executable_provider,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    int64_t num_replicas, bool run_hlo_passes,
    DeviceAssignment* device_assignment) {
  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;
  return test_runner_->ExecuteReplicated(
      executable_provider, argument_count_provider, argument_provider,
      std::move(options), device_assignment);
}

absl::StatusOr<std::vector<Literal>>
HloRunnerAgnosticTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module,
    std::vector<std::vector<Literal*>> arguments, int64_t num_replicas,
    bool run_hlo_passes, DeviceAssignment* device_assignment) {
  CHECK_GT(num_replicas, 0) << "expect at least one replica";
  CHECK_EQ(num_replicas, arguments.size())
      << "expect arguments for each replica";
  int64_t argument_count = arguments.front().size();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<OpaqueExecutable> executable,
      test_runner_->CreateExecutable(std::move(module), run_hlo_passes));
  return ExecuteReplicated(
      /*executable_provider=*/[&](int64_t) { return executable.get(); },
      /*argument_count_provider=*/[&](int64_t) { return argument_count; },
      /*argument_provider=*/
      [&](int64_t replica_idx, int64_t argument_idx) -> const Literal* {
        return arguments[replica_idx][argument_idx];
      },
      num_replicas, /*run_hlo_passes=*/run_hlo_passes,
      /*device_assignment=*/device_assignment);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::Run(
    std::unique_ptr<HloModule> module, bool run_hlo_passes,
    const std::function<void(HloModule*)>& test_preprocessor) {
  std::vector<Literal> fake_arguments = MakeFakeArguments(module.get()).value();
  if (absl::StatusOr<bool> change = verifier().Run(module.get());
      !change.ok()) {
    return ::testing::AssertionFailure() << change.status();
  }
  if (absl::Status status = PreprocessModuleForTestRunner(module.get());
      !status.ok()) {
    return ::testing::AssertionFailure() << status;
  }
  if (test_preprocessor != nullptr) {
    test_preprocessor(module.get());
  }

  absl::StatusOr<Literal> output =
      test_runner_->Execute(std::move(module), fake_arguments, run_hlo_passes);
  return output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().message();
}

::testing::AssertionResult
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const HloRunnerInterface::ReplicatedExecuteOptions& options) {
  const int replica_count = module_0->config().replica_count();
  if (replica_count != module_1->config().replica_count()) {
    return ::testing::AssertionFailure()
           << "Number of replicas is not the same: " << replica_count << " Vs "
           << module_1->config().replica_count();
  }
  if (options.num_replicas != replica_count) {
    return ::testing::AssertionFailure()
           << "Number of execution replicas is different from number of "
              "replicas in the module: requested number of replicas = "
           << options.num_replicas
           << ", number of replicas in hlo = " << replica_count;
  }

  if (std::vector<int> mismatches = CompareInputs(*module_0, *module_1);
      !mismatches.empty()) {
    return ::testing::AssertionFailure()
           << "Error: parameter mismatch at indices: "
           << absl::StrJoin(mismatches, ",");
  }
  if (int64_t num_args = module_0->entry_computation()->num_parameters();
      num_args != options.arguments.size()) {
    return ::testing::AssertionFailure()
           << "Mismatch in number of arguments passed while running replicated "
              "hlo module. Expected: "
           << num_args << ", actual: " << options.arguments.size();
  }
  absl::StatusOr<::testing::AssertionResult> result =
      RunAndCompareTwoModulesInternalReplicated(std::move(module_0),
                                                std::move(module_1), options);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return *result;
}

::testing::AssertionResult
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    bool run_hlo_passes, bool use_threads) {
  absl::StatusOr<std::vector<Literal>> fake_arguments = MakeFakeArguments(
      /*module=*/module_0.get(), /*pseudo_random=*/true,
      /*treat_gte_as_data_formatting=*/false,
      /*max_bits_of_precision=*/std::nullopt);
  CHECK_OK(fake_arguments);

  return RunAndCompareTwoModulesReplicated(std::move(module_0),
                                           std::move(module_1), *fake_arguments,
                                           run_hlo_passes, use_threads);
}

::testing::AssertionResult
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const std::vector<Literal>& fake_arguments, bool run_hlo_passes,
    bool use_threads) {
  std::vector<const Literal*> fake_argument_ptrs;
  absl::c_transform(
      /*input=*/fake_arguments,
      /*output=*/std::back_inserter(fake_argument_ptrs),
      /*unary_op=*/[](const Literal& literal) -> const Literal* {
        return &literal;
      });
  HloRunnerInterface::ReplicatedExecuteOptions options{
      /*num_replicas=*/module_0->config().replica_count(),
      /*arguments=*/fake_argument_ptrs,
      /*infeed_values=*/{},
      /*infeed_steps=*/-1,
      /*outfeed_shape=*/{},
      /*outfeed_values=*/nullptr,
      /*run_hlo_passes=*/run_hlo_passes,
      /*use_threads=*/use_threads};
  return RunAndCompareTwoModulesReplicated(std::move(module_0),
                                           std::move(module_1), options);
}

::testing::AssertionResult
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesReplicated(
    std::string_view module_0_str, std::string_view module_1_str,
    bool run_hlo_passes, bool use_threads) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_0 =
      ParseAndReturnVerifiedModule(module_0_str);
  if (!module_0.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0.status().ToString();
  }

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_1 =
      ParseAndReturnVerifiedModule(module_1_str);
  if (!module_1.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1.status().ToString();
  }
  return RunAndCompareTwoModulesReplicated(
      *std::move(module_0), *std::move(module_1), run_hlo_passes, use_threads);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareTwoModules(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    absl::Span<const Literal* const> arguments, bool run_hlo_passes) {
  absl::StatusOr<::testing::AssertionResult> result =
      RunAndCompareTwoModulesInternal(std::move(module_0), std::move(module_1),
                                      arguments, run_hlo_passes);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return *result;
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareTwoModules(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    bool run_hlo_passes, std::optional<int64_t> args_max_bits_of_precision) {
  if (std::vector<int> mismatches = CompareInputs(*module_0, *module_1);
      !mismatches.empty()) {
    return ::testing::AssertionFailure()
           << "Error : mismatching parameter shapes for parameters "
           << absl::StrJoin(mismatches, ", ");
  }

  absl::StatusOr<std::vector<Literal>> fake_arguments = MakeFakeArguments(
      module_0.get(), /*pseudo_random=*/true,
      /*treat_gte_as_data_formatting=*/false, args_max_bits_of_precision);
  CHECK_OK(fake_arguments);

  std::vector<const Literal*> fake_argument_ptrs;
  absl::c_transform(*fake_arguments, std::back_inserter(fake_argument_ptrs),
                    [](const Literal& literal) { return &literal; });

  return RunAndCompareTwoModules(std::move(module_0), std::move(module_1),
                                 fake_argument_ptrs, run_hlo_passes);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareTwoModules(
    std::string_view hlo_string_module_0, std::string_view hlo_string_module_1,
    bool run_hlo_passes, std::optional<int64_t> args_max_bits_of_precision) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_0 =
      ParseAndReturnVerifiedModule(hlo_string_module_0);
  if (!module_0.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0.status().ToString();
  }

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_1 =
      ParseAndReturnVerifiedModule(hlo_string_module_1);
  if (!module_1.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1.status().ToString();
  }
  return RunAndCompareTwoModules(*std::move(module_0), *std::move(module_1),
                                 run_hlo_passes, args_max_bits_of_precision);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareTwoModules(
    std::string_view hlo_string_module_0, std::string_view hlo_string_module_1,
    const HloModuleConfig& config_0, const HloModuleConfig& config_1,
    bool run_hlo_passes, std::optional<int64_t> args_max_bits_of_precision) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_0 =
      ParseAndReturnVerifiedModule(hlo_string_module_0, config_0);
  if (!module_0.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0.status().ToString();
  }

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module_1 =
      ParseAndReturnVerifiedModule(hlo_string_module_1, config_1);
  if (!module_1.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1.status().ToString();
  }
  return RunAndCompareTwoModules(*std::move(module_0), *std::move(module_1),
                                 run_hlo_passes, args_max_bits_of_precision);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunAndCompareTwoModules(
    std::string_view hlo_string_module_0, std::string_view hlo_string_module_1,
    absl::Span<const Literal* const> arguments, bool run_hlo_passes) {
  auto module_0_or_status = ParseAndReturnVerifiedModule(hlo_string_module_0);
  if (!module_0_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0_or_status.status().ToString();
  }

  auto module_1_or_status = ParseAndReturnVerifiedModule(hlo_string_module_1);
  if (!module_1_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1_or_status.status().ToString();
  }
  return RunAndCompareTwoModules(std::move(module_0_or_status).value(),
                                 std::move(module_1_or_status).value(),
                                 arguments, run_hlo_passes);
}

::testing::AssertionResult HloRunnerAgnosticTestBase::Run(
    std::string_view hlo_string, bool run_hlo_passes, ExecutionProfile* profile,
    const google::protobuf::Message* backend_config, bool use_random_data) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
      ParseAndReturnVerifiedModule(hlo_string);
  if (!module.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module.status().ToString();
  }
  if (absl::Status status = PreprocessModuleForTestRunner(module->get());
      !status.ok()) {
    return ::testing::AssertionFailure() << status;
  }
  const std::vector<Literal> fake_arguments =
      MakeFakeArguments(module->get(), use_random_data).value();
  std::vector<const Literal*> fake_argument_ptrs;
  absl::c_transform(fake_arguments, std::back_inserter(fake_argument_ptrs),
                    [](const Literal& literal) { return &literal; });

  if (profile != nullptr) {
    // We have to enable HLO profiling since otherwise currently the
    // ExecutionProfile is not correct.
    HloModuleConfig config = (*module)->config();
    DebugOptions debug_options = config.debug_options();
    debug_options.set_zkx_hlo_profile(true);
    config.set_debug_options(debug_options);
    (*module)->set_config(config);
  }

  if (backend_config) {
    // Set backend configuration if it is given.
    HloInstruction* instruction =
        (*module)->entry_computation()->root_instruction();
    absl::Status s = instruction->set_backend_config(*backend_config);
    return s.ok() ? ::testing::AssertionSuccess()
                  : ::testing::AssertionFailure() << s.message();
  }

  auto output = test_runner_->Execute(*std::move(module), fake_argument_ptrs,
                                      /*run_hlo_passes=*/run_hlo_passes,
                                      /*profile=*/profile);

  return output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().message();
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunReplicated(
    std::string_view hlo_string, bool run_hlo_passes, int64_t num_replicas,
    const google::protobuf::Message* backend_config) {
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
      ParseAndReturnVerifiedModule(hlo_string, num_replicas);
  if (!module.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module.status().ToString();
  }

  std::vector<Literal> fake_arguments =
      MakeFakeArguments(module->get()).value();
  std::vector<const Literal*> fake_argument_ptrs;
  absl::c_transform(fake_arguments, std::back_inserter(fake_argument_ptrs),
                    [](const Literal& literal) { return &literal; });

  if (backend_config) {
    // Set backend configuration if it is given.
    HloInstruction* instruction =
        (*module)->entry_computation()->root_instruction();
    if (absl::Status s = instruction->set_backend_config(*backend_config);
        !s.ok()) {
      return ::testing::AssertionFailure() << s.message();
    }
    return ::testing::AssertionSuccess();
  }

  HloRunnerInterface::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.arguments = {fake_argument_ptrs.begin(), fake_argument_ptrs.end()};
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;
  absl::StatusOr<std::vector<Literal>> output =
      test_runner_->ExecuteReplicated(*std::move(module), std::move(options));
  if (output.ok()) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << output.status().message();
}

::testing::AssertionResult HloRunnerAgnosticTestBase::RunMultipleTimes(
    std::string_view hlo_string, bool run_hlo_passes,
    std::vector<ExecutionProfile>* profiles,
    const google::protobuf::Message* backend_config, bool assert_determinism) {
  int n = profiles->size();
  std::vector<std::vector<Literal>> fake_arguments(n);
  std::vector<std::unique_ptr<OpaqueExecutable>> executables(n);

  for (int i = 0; i < n; ++i) {
    absl::StatusOr<std::unique_ptr<VerifiedHloModule>> module =
        ParseAndReturnVerifiedModule(hlo_string);
    if (!module.ok()) {
      return ::testing::AssertionFailure()
             << "Error while parsing HLO text format: "
             << module.status().ToString();
    }

    fake_arguments[i] = MakeFakeArguments(module->get()).value();

    if (profiles != nullptr) {
      // We have to enable HLO profiling since otherwise currently the
      // ExecutionProfile is not correct.
      HloModuleConfig config = (*module)->config();
      DebugOptions debug_options = config.debug_options();
      debug_options.set_zkx_hlo_profile(true);
      config.set_debug_options(debug_options);
      (*module)->set_config(config);
    }

    if (backend_config) {
      // Set backend configuration if it is given.
      HloInstruction* instruction =
          (*module)->entry_computation()->root_instruction();
      absl::Status s = instruction->set_backend_config(*backend_config);
      return s.ok() ? ::testing::AssertionSuccess()
                    : ::testing::AssertionFailure() << s.message();
    }

    absl::StatusOr<std::unique_ptr<OpaqueExecutable>> executable =
        test_runner_->CreateExecutable(*std::move(module), run_hlo_passes);
    if (!executable.ok()) {
      return ::testing::AssertionFailure() << executable.status().message();
    }
    executables[i] = *std::move(executable);
  }

  std::optional<Literal> canonical_output;
  for (int i = 0; i < n; ++i) {
    absl::StatusOr<Literal> output = test_runner_->ExecuteWithExecutable(
        executables[i].get(), fake_arguments[i],
        /*profile=*/&((*profiles)[i]));
    if (!output.ok()) {
      return ::testing::AssertionFailure() << output.status().message();
    }

    if (assert_determinism) {
      if (!canonical_output.has_value()) {
        canonical_output = *std::move(output);
      } else {
        if (*canonical_output != *output) {
          return ::testing::AssertionFailure()
                 << "Successive runs have returned different results: "
                 << *canonical_output << " vs. " << *output;
        }
      }
    }
  }

  return ::testing::AssertionSuccess();
}

absl::StatusOr<::testing::AssertionResult>
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesInternalReplicated(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const HloRunnerInterface::ReplicatedExecuteOptions& options) {
  TF_RETURN_IF_ERROR(verifier().Run(module_0.get()).status());
  TF_RETURN_IF_ERROR(verifier().Run(module_1.get()).status());

  // Execute the two modules.
  TF_ASSIGN_OR_RETURN(auto test_0, test_runner_->ExecuteReplicated(
                                       std::move(module_0), options));
  TF_ASSIGN_OR_RETURN(auto test_1, test_runner_->ExecuteReplicated(
                                       std::move(module_1), options));

  for (const auto& [expected, actual] : llvm::zip_equal(test_0, test_1)) {
    if (::testing::AssertionResult result =
            LiteralTestUtil::Equal(expected, actual);
        !result) {
      return result;
    }
  }
  return ::testing::AssertionSuccess();
}

absl::StatusOr<::testing::AssertionResult>
HloRunnerAgnosticTestBase::RunAndCompareTwoModulesInternal(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    absl::Span<const Literal* const> arguments, bool run_hlo_passes) {
  TF_RETURN_IF_ERROR(verifier().Run(module_0.get()).status());
  TF_RETURN_IF_ERROR(verifier().Run(module_1.get()).status());

  // Execute the two modules.
  TF_ASSIGN_OR_RETURN(
      const Literal test_0,
      test_runner_->Execute(std::move(module_0), arguments, run_hlo_passes));
  TF_ASSIGN_OR_RETURN(
      const Literal test_1,
      test_runner_->Execute(std::move(module_1), arguments, run_hlo_passes));

  return LiteralTestUtil::Equal(/*expected=*/test_0, /*actual=*/test_1);
}

}  // namespace zkx
