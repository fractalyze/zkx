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

#include "xla/tsl/platform/status.h"

#include <sstream>

namespace tsl {

std::string* TfCheckOpHelperOutOfLine(const absl::Status& v, const char* msg) {
  std::stringstream ss;
  ss << "Non-OK-status: " << msg << "\nStatus: " << v;

  // Leaks string but this is only to be used in a fatal error message
  return new std::string(ss.str());
}

}  // namespace tsl
