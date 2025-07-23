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
