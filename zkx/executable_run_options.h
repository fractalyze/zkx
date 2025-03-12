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

#ifndef ZKX_EXECUTABLE_RUN_OPTIONS_H_
#define ZKX_EXECUTABLE_RUN_OPTIONS_H_

#include <stdint.h>

#include <string>

#include "absl/hash/hash.h"

namespace zkx {

// A unique identifier for a particular "logical execution" of an ZKX model.
//
// A logical execution might encompass multiple executions of one or more
// HloModules.  Runs that are part of the same logical execution can
// communicate via collective ops (e.g. kAllToAll), whereas runs that are part
// of different logical executions are isolated.
class RunId {
 public:
  // Creates a new, unique RunId.
  RunId();
  explicit RunId(int64_t value) : data_(value) {}

  RunId(const RunId&) = default;
  RunId& operator=(const RunId&) = default;
  bool operator==(const RunId& other) const { return data_ == other.data_; }
  std::string ToString() const;
  int64_t ToInt() const;

  template <typename H>
  friend H AbslHashValue(H h, const RunId& id) {
    return H::combine(std::move(h), id.data_);
  }

 private:
  int64_t data_;
};

}  // namespace zkx

#endif  // ZKX_EXECUTABLE_RUN_OPTIONS_H_
