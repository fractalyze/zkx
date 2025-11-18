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

#ifndef ZKX_BASE_TEST_SCOPED_ENVIRONMENT_H_
#define ZKX_BASE_TEST_SCOPED_ENVIRONMENT_H_

#include <optional>
#include <string>

namespace zkx::base {

class ScopedEnvironment {
 public:
  ScopedEnvironment(std::string_view env_name);
  ScopedEnvironment(std::string_view env_name, std::string_view value,
                    bool overwrite);
  ~ScopedEnvironment();

  ScopedEnvironment(const ScopedEnvironment&) = delete;
  ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;
  ScopedEnvironment(ScopedEnvironment&&) = delete;
  ScopedEnvironment& operator=(ScopedEnvironment&&) = delete;

 private:
  std::string env_name_;
  std::optional<std::string> env_value_;
};

}  // namespace zkx::base

#endif  // ZKX_BASE_TEST_SCOPED_ENVIRONMENT_H_
