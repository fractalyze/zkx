#include "zkx/base/test/scoped_environment.h"

#include <cstdlib>

#include "gtest/gtest.h"

#include "xla/tsl/platform/env.h"

namespace zkx::base {
namespace {

TEST(ScopedEnvironmentTest, RestoresOriginalValueEvenIfModifiedInsideScope) {
  const char* var_name = "ZKX_SCOPED_ENV_TEST_VAR";
  tsl::setenv(var_name, "foo", /*overwrite=*/true);

  {
    ScopedEnvironment scoped_env(var_name);
    tsl::setenv(var_name, "bar", /*overwrite=*/true);
  }
  const char* restored = std::getenv(var_name);
  EXPECT_STREQ(restored, "foo");

  tsl::unsetenv(var_name);
}

TEST(ScopedEnvironmentTest, SetsAndRestoresEnvironmentVariable) {
  const char* var_name = "ZKX_SCOPED_ENV_TEST_VAR2";
  // Ensure the variable is not set initially
  tsl::unsetenv(var_name);

  {
    ScopedEnvironment scoped_env(var_name, "foo", /*overwrite=*/true);
    const char* value = std::getenv(var_name);
    EXPECT_STREQ(value, "foo");
  }
  // After destruction, variable should be unset
  EXPECT_EQ(std::getenv(var_name), nullptr);
}

TEST(ScopedEnvironmentTest, SetThenRestoreExistingVariable) {
  const char* var_name = "ZKX_SCOPED_ENV_TEST_VAR3";
  tsl::setenv(var_name, "original", /*overwrite=*/true);

  {
    ScopedEnvironment scoped_env(var_name, "bar", /*overwrite=*/true);
    const char* value = std::getenv(var_name);
    EXPECT_STREQ(value, "bar");
  }
  // After destruction, variable should be restored to original value
  const char* restored = std::getenv(var_name);
  EXPECT_STREQ(restored, "original");

  tsl::unsetenv(var_name);
}

TEST(ScopedEnvironmentTest, DoesNotOverwriteWhenFlagIsFalse) {
  const char* var_name = "ZKX_SCOPED_ENV_TEST_VAR4";
  tsl::setenv(var_name, "existing", /*overwrite=*/true);

  {
    ScopedEnvironment scoped_env(var_name, "newvalue", /*overwrite=*/false);
    const char* value = std::getenv(var_name);
    EXPECT_STREQ(value, "existing");
  }
  const char* restored = std::getenv(var_name);
  EXPECT_STREQ(restored, "existing");

  tsl::unsetenv(var_name);
}

}  // namespace
}  // namespace zkx::base
