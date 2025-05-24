#include "zkx/base/containers/container_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace zkx::base {

TEST(ContainerUtilTest, Map) {
  std::vector<int> arr({1, 2, 3});
  EXPECT_THAT(Map(arr.begin(), arr.end(),
                  [](int v) { return static_cast<double>(v * 2); }),
              testing::ContainerEq(std::vector<double>{2.0, 4.0, 6.0}));
  EXPECT_THAT(Map(arr, [](int v) { return static_cast<double>(v * 2); }),
              testing::ContainerEq(std::vector<double>{2.0, 4.0, 6.0}));
}

TEST(ContainerUtilTest, MapWithIdx) {
  std::vector<int> arr({1, 2, 3});
  EXPECT_THAT(
      Map(arr.begin(), arr.end(),
          [](size_t idx, int v) { return static_cast<double>(v * 2 + idx); }),
      testing::ContainerEq(std::vector<double>{2.0, 5.0, 8.0}));
  EXPECT_THAT(Map(arr, [](size_t idx,
                          int v) { return static_cast<double>(v * 2 + idx); }),
              testing::ContainerEq(std::vector<double>{2.0, 5.0, 8.0}));
}

}  // namespace zkx::base
