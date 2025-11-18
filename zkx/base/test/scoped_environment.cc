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

#include "zkx/base/test/scoped_environment.h"

#include <cstdlib>

#include "xla/tsl/platform/env.h"

namespace zkx::base {

ScopedEnvironment::ScopedEnvironment(std::string_view env_name)
    : env_name_(env_name) {
  const char* env_value = std::getenv(env_name_.c_str());
  if (env_value) {
    env_value_ = env_value;
  }
}

ScopedEnvironment::ScopedEnvironment(std::string_view env_name,
                                     std::string_view value, bool overwrite)
    : ScopedEnvironment(env_name) {
  tsl::setenv(env_name_.c_str(), std::string(value).c_str(), overwrite);
}

ScopedEnvironment::~ScopedEnvironment() {
  if (env_value_) {
    tsl::setenv(env_name_.c_str(), env_value_->c_str(), /*overwrite=*/true);
  } else {
    tsl::unsetenv(env_name_.c_str());
  }
}

}  // namespace zkx::base
