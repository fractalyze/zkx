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

#include "zkx/base/strings/string_util.h"

#include "gtest/gtest.h"

namespace zkx::base {

TEST(HumanReadableElapsedTime, Basic) {
  EXPECT_EQ(HumanReadableElapsedTime(-10), "-10 s");
  EXPECT_EQ(HumanReadableElapsedTime(-0.001), "-1 ms");
  EXPECT_EQ(HumanReadableElapsedTime(-60.0), "-1 min");
  EXPECT_EQ(HumanReadableElapsedTime(0.00000001), "0.01 us");
  EXPECT_EQ(HumanReadableElapsedTime(0.0000012), "1.2 us");
  EXPECT_EQ(HumanReadableElapsedTime(0.0012), "1.2 ms");
  EXPECT_EQ(HumanReadableElapsedTime(0.12), "120 ms");
  EXPECT_EQ(HumanReadableElapsedTime(1.12), "1.12 s");
  EXPECT_EQ(HumanReadableElapsedTime(90.0), "1.5 min");
  EXPECT_EQ(HumanReadableElapsedTime(600.0), "10 min");
  EXPECT_EQ(HumanReadableElapsedTime(9000.0), "2.5 h");
  EXPECT_EQ(HumanReadableElapsedTime(87480.0), "1.01 days");
  EXPECT_EQ(HumanReadableElapsedTime(7776000.0), "2.96 months");
  EXPECT_EQ(HumanReadableElapsedTime(78840000.0), "2.5 years");
  EXPECT_EQ(HumanReadableElapsedTime(382386614.40), "12.1 years");
  EXPECT_EQ(HumanReadableElapsedTime(DBL_MAX), "5.7e+300 years");
}

}  // namespace zkx::base
