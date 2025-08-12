/* Copyright 2019 The OpenXLA Authors.

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
#ifndef ZKX_HLO_TESTLIB_VERIFIED_HLO_MODULE_H_
#define ZKX_HLO_TESTLIB_VERIFIED_HLO_MODULE_H_

#include <stdint.h>

#include <functional>
#include <string>

#include "absl/status/status.h"

#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/service/hlo_verifier.h"
#include "zkx/shape.h"
#include "zkx/util.h"

namespace zkx {

// An HLO module derived class which verifies itself on destruction. This class
// is intended to be used in unit tests. Any verification errors are raised via
// ADD_FAILURE.
class VerifiedHloModule : public HloModule {
 public:
  VerifiedHloModule(const std::string& name, const HloModuleConfig& config,
                    bool verifier_layout_sensitive,
                    std::function<int64_t(const Shape&)> shape_size_function,
                    HloPredicate instruction_can_change_layout_func = {})
      : HloModule(name, config),
        verifier_(verifier_layout_sensitive, instruction_can_change_layout_func,
                  shape_size_function) {}

  ~VerifiedHloModule() override { VerifyOrAddFailure("in destructor"); }

  // Given a string in the HloModule::ToString() format, parses the string and
  // builds the VerifiedHloModule in place. Before calling this method, the
  // module must be empty (no computations). Finally verifies the module using
  // HloVerifier and returns the status.
  absl::Status ParseHloStringAndVerifyModule(
      std::string_view str,
      const HloParserOptions& options = HloParserOptions());

  // Verifies the module and flags any error with ADD_FAILURE. 'message' is
  // included in the failure message.
  void VerifyOrAddFailure(std::string_view message);

  // Verifies the module using HloVerifier and returns the status.
  absl::Status Verify();

 private:
  HloVerifier verifier_;
};

}  // namespace zkx

#endif  // ZKX_HLO_TESTLIB_VERIFIED_HLO_MODULE_H_
