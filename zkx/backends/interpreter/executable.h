/* Copyright 2017 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#ifndef ZKX_BACKENDS_INTERPRETER_EXECUTABLE_H_
#define ZKX_BACKENDS_INTERPRETER_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

#include "zkx/backends/interpreter/executable_base.h"
#include "zkx/hlo/evaluator/hlo_evaluator.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/service/dynamic_dimension_inference.h"

namespace zkx::interpreter {

// Responsible for running a HLO graph through the HloEvaluator and output
// buffer allocation. Refer to interpreter/README.md for more.
class InterpreterExecutable : public InterpreterExecutableBase {
 public:
  InterpreterExecutable(
      std::unique_ptr<HloModule> hlo_module,
      std::unique_ptr<HloEvaluator> evaluator,
      std::optional<DynamicDimensionInference> dynamic_dimension_inference);

  static int64_t ShapeSizeBytes(const Shape& shape);

 protected:
  absl::StatusOr<Literal> Evaluate(
      const ServiceExecutableRunOptions* run_options,
      const HloComputation& computation,
      absl::Span<const Literal> arg_literals) override
      ABSL_LOCKS_EXCLUDED(evaluator_lock_);

  // The interpreter interprets executables with an HloEvaluator.
  std::unique_ptr<HloEvaluator> evaluator_ ABSL_PT_GUARDED_BY(evaluator_lock_);
  mutable absl::Mutex evaluator_lock_;

 private:
  std::optional<DynamicDimensionInference> dynamic_dimension_inference_;
  InterpreterExecutable(const InterpreterExecutable&) = delete;
  InterpreterExecutable& operator=(const InterpreterExecutable&) = delete;
};

}  // namespace zkx::interpreter

#endif  // ZKX_BACKENDS_INTERPRETER_EXECUTABLE_H_
