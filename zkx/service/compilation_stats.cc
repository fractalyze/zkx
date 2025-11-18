/* Copyright 2019 The OpenXLA Authors.
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

#include "zkx/service/compilation_stats.h"

#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"

#include "xla/tsl/platform/env.h"

namespace zkx {
namespace {

class NoopStats : public CompilationStats {
 public:
  NoopStats() = default;

  void StartPass(std::string_view pass_name) override {}

  void EndPass(std::string_view pass_name) override {}

  void CompilationReport() const override {}

  size_t GetPassesSize() const override { return 0; }

  void RecordPassError(std::string_view pass_name,
                       std::string_view err) override {};
};

class Stats : public CompilationStats {
 public:
  Stats() = default;

  void StartPass(std::string_view pass_name) override;

  void EndPass(std::string_view pass_name) override;

  void CompilationReport() const override;

  size_t GetPassesSize() const override;

  void RecordPassError(std::string_view pass_name,
                       std::string_view err) override {};

 private:
  struct PassInfo {
    PassInfo(std::string_view name, double duration)
        : name(name), duration_ms(duration) {}

    std::string name;
    int num_runs = 1;
    double duration_ms;
  };

  // Info about the passes that have been run so far.
  std::vector<PassInfo> passes_;
  // Used to avoid nested calls to StartPass.
  bool pass_running_ = false;
  std::string current_pass_;
  // The start time of the currently running pass.
  uint64_t start_micros_;
};

void Stats::StartPass(std::string_view pass_name) {
  CHECK(!pass_running_) << "Can't start " << pass_name << " while running "
                        << current_pass_;
  pass_running_ = true;
  current_pass_ = std::string(pass_name);
  start_micros_ = tsl::Env::Default()->NowMicros();
}

void Stats::EndPass(std::string_view pass_name) {
  CHECK(pass_running_);
  CHECK_EQ(current_pass_, pass_name);
  pass_running_ = false;
  uint64_t end_micros = tsl::Env::Default()->NowMicros();
  double duration_ms = (end_micros - start_micros_) / 1000.0;
  passes_.push_back(PassInfo(current_pass_, duration_ms));
}

void Stats::CompilationReport() const {
  CHECK(!pass_running_) << "EndPass never called for " << current_pass_;
  absl::flat_hash_map<std::string, PassInfo> summary;
  double total_duration = 0;

  for (const PassInfo& pass_run : passes_) {
    std::string_view pass_name = pass_run.name;
    total_duration += pass_run.duration_ms;
    auto it = summary.find(pass_name);
    if (it == summary.end()) {
      summary.emplace(pass_name, pass_run);
    } else {
      ++summary.at(pass_name).num_runs;
      summary.at(pass_name).duration_ms += pass_run.duration_ms;
    }
  }

  std::vector<PassInfo> sorted_summary;
  sorted_summary.reserve(summary.size());
  for (auto& it : summary) {
    sorted_summary.push_back(it.second);
  }
  absl::c_sort(sorted_summary, [](const PassInfo& a, const PassInfo& b) {
    // Sort passes that take the longest first, break ties using pass names.
    return std::make_pair(b.duration_ms, a.name) <
           std::make_pair(a.duration_ms, b.name);
  });
  LOG(INFO) << "Total runtime (ms) of HLO passes: " << total_duration;
  LOG(INFO) << "Pass name, num runs, time (ms)";
  for (const PassInfo& pass_info : sorted_summary) {
    LOG(INFO) << pass_info.name << ", " << pass_info.num_runs << ", "
              << pass_info.duration_ms;
  }
}

size_t Stats::GetPassesSize() const { return passes_.size(); }

}  // namespace

// static
std::unique_ptr<CompilationStats> CompilationStats::MakeNoopStats() {
  return std::make_unique<NoopStats>();
}

// static
std::unique_ptr<CompilationStats> CompilationStats::MakeStats() {
  return std::make_unique<Stats>();
}

}  // namespace zkx
