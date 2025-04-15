/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/platform/path.h"

#include <stdlib.h>

#include "absl/strings/str_cat.h"

namespace tsl::io {
namespace internal {
namespace {

const char kPathSep[] = "/";

}  // namespace

std::string JoinPathImpl(std::initializer_list<std::string_view> paths) {
  std::string result;

  for (std::string_view path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = std::string(path);
      continue;
    }

    if (IsAbsolutePath(path)) path = path.substr(1);

    if (result[result.size() - 1] == kPathSep[0]) {
      absl::StrAppend(&result, path);
    } else {
      absl::StrAppend(&result, kPathSep, path);
    }
  }

  return result;
}

}  // namespace internal

bool IsAbsolutePath(std::string_view path) {
  return !path.empty() && path[0] == '/';
}

bool GetTestUndeclaredOutputsDir(std::string* dir) {
  const char* outputs_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (outputs_dir == nullptr) {
    return false;
  }
  if (dir != nullptr) {
    *dir = outputs_dir;
  }
  return true;
}

}  // namespace tsl::io
