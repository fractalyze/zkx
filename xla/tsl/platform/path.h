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

#include <string>

namespace tsl::io {

// Returns whether the TEST_UNDECLARED_OUTPUTS_DIR environment variable is set.
// If it's set and dir != nullptr then sets *dir to that.
bool GetTestUndeclaredOutputsDir(std::string* dir);

}  // namespace tsl::io

#endif  // XLA_TSL_PLATFORM_PATH_H_
