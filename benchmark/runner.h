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

#ifndef BENCHMARK_RUNNER_H_
#define BENCHMARK_RUNNER_H_

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/base/buffer/read_only_buffer.h"
#include "zkx/base/flag/flag_parser.h"
#include "zkx/hlo/parser/hlo_parser.h"
#include "zkx/literal.h"
#include "zkx/service/hlo_runner.h"

namespace zkx::benchmark {

class Runner {
 public:
  struct Options {
    bool multi_degrees = false;
  };

  Runner();
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  base::FlagParser& parser() { return parser_; }

  void AddPositionalFlags();
  void AddOptionalFlags(const Options& options);
  absl::Status Parse(int argc, char** argv);

  uint32_t GetMaxDegree() const {
    CHECK(!degrees_.empty());
    return degrees_.back();
  }
  size_t GetMaxSize() const { return size_t{1} << GetMaxDegree(); }

  template <typename F>
  absl::Status Run(std::function<std::string(uint32_t)> hlo_generator,
                   std::function<std::vector<Literal>(absl::Span<const F>)>
                       literals_generator) {
    std::vector<F> input;
    {
      std::string input_string;
      TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), input_path_,
                                               &input_string));

      base::ReadOnlyBuffer buffer(input_string.data(), input_string.length());

      TF_RETURN_IF_ERROR(buffer.Read(&input));
    }

    size_t max_size = GetMaxSize();
    if (max_size > input.size()) {
      return absl::OutOfRangeError(
          absl::Substitute("Input file contains only $0 scalars, need $1",
                           input.size(), max_size));
    }

    durations_.resize(degrees_.size());

    for (size_t i = 0; i < degrees_.size(); ++i) {
      uint32_t degree = degrees_[i];
      std::cout << "Running with degree: " << degree << std::endl;

      std::string hlo_string = hlo_generator(degree);
      TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
      TF_ASSIGN_OR_RETURN(std::unique_ptr<OpaqueExecutable> executable,
                          runner_.CreateExecutable(std::move(module),
                                                   /*run_hlo_passes=*/false));

      std::vector<Literal> literals = literals_generator(
          absl::MakeConstSpan(input).subspan(0, size_t{1} << degree));
      Literal output;
      for (uint32_t j = 0; j < num_warmups_; ++j) {
        TF_ASSIGN_OR_RETURN(
            output, runner_.ExecuteWithExecutable(executable.get(),
                                                  absl::MakeConstSpan(literals),
                                                  /*profile=*/nullptr));
      }
      for (uint32_t j = 0; j < num_runs_; ++j) {
        auto start_time = absl::Now();
        TF_ASSIGN_OR_RETURN(
            output, runner_.ExecuteWithExecutable(executable.get(),
                                                  absl::MakeConstSpan(literals),
                                                  /*profile=*/nullptr));
        auto end_time = absl::Now();
        durations_[i].push_back(end_time - start_time);
      }
    }
    return absl::OkStatus();
  }

  void PrintResults() const;

  absl::Status MaybeWriteCsv() const;

 private:
  HloRunner runner_;
  base::FlagParser parser_;
  std::string input_path_;
  std::vector<uint32_t> degrees_;
  uint32_t num_warmups_;
  uint32_t num_runs_;
  std::string output_path_;
  std::vector<std::vector<absl::Duration>> durations_;
};

}  // namespace zkx::benchmark

#endif  // BENCHMARK_RUNNER_H_
