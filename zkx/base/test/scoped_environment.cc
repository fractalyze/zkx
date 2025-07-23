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
