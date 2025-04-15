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

#ifndef XLA_TSL_PLATFORM_PATH_H_
#define XLA_TSL_PLATFORM_PATH_H_

#include <initializer_list>
#include <string>

namespace tsl::io {
namespace internal {
std::string JoinPathImpl(std::initializer_list<std::string_view> paths);
}

// Utility routines for processing filenames

#ifndef SWIG  // variadic templates
// Join multiple paths together, without introducing unnecessary path
// separators.
// For example:
//
//  Arguments                  | JoinPath
//  ---------------------------+----------
//  '/foo', 'bar'              | /foo/bar
//  '/foo/', 'bar'             | /foo/bar
//  '/foo', '/bar'             | /foo/bar
//
// Usage:
// string path = io::JoinPath("/mydir", filename);
// string path = io::JoinPath(FLAGS_test_srcdir, filename);
// string path = io::JoinPath("/full", "path", "to", "filename");
template <typename... T>
std::string JoinPath(const T&... args) {
  return internal::JoinPathImpl({args...});
}
#endif /* SWIG */

// Return true if path is absolute.
bool IsAbsolutePath(std::string_view path);

// Returns whether the TEST_UNDECLARED_OUTPUTS_DIR environment variable is set.
// If it's set and dir != nullptr then sets *dir to that.
bool GetTestUndeclaredOutputsDir(std::string* dir);

}  // namespace tsl::io

#endif  // XLA_TSL_PLATFORM_PATH_H_
