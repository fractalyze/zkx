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

#ifndef ZKX_STREAM_EXECUTOR_DNN_H_
#define ZKX_STREAM_EXECUTOR_DNN_H_

#include <string>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "zkx/stream_executor/namespace_alias.h"

namespace stream_executor {
namespace dnn {

// Represents a version of a DNN library.
class VersionInfo {
 public:
  constexpr VersionInfo(int major = 0, int minor = 0, int patch = 0)
      : major_(major), minor_(minor), patch_(patch) {}

  int major_version() const { return major_; }
  int minor_version() const { return minor_; }
  int patch() const { return patch_; }

  std::tuple<int, int, int> as_tuple() const {
    return std::make_tuple(major_, minor_, patch_);
  }

  std::string ToString() const {
    return absl::StrCat(major_, ".", minor_, ".", patch_);
  }

  friend bool operator<(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() < b.as_tuple();
  }
  friend bool operator<=(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() <= b.as_tuple();
  }
  friend bool operator>(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() > b.as_tuple();
  }
  friend bool operator>=(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() >= b.as_tuple();
  }
  friend bool operator==(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() == b.as_tuple();
  }
  friend bool operator!=(const VersionInfo& a, const VersionInfo& b) {
    return a.as_tuple() != b.as_tuple();
  }

 private:
  int major_;
  int minor_;
  int patch_;
};

// DnnSupport interface - main DNN operations interface.
// This is the base class for platform-specific DNN implementations.
class DnnSupport {
 public:
  virtual ~DnnSupport() = default;

  virtual absl::Status Init() { return absl::OkStatus(); }

  virtual absl::StatusOr<VersionInfo> GetVersion() {
    return absl::UnimplementedError("DNN not supported in zkx");
  }
};

}  // namespace dnn
}  // namespace stream_executor

#endif  // ZKX_STREAM_EXECUTOR_DNN_H_
