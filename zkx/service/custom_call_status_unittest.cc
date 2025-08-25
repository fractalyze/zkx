/* Copyright 2021 The OpenXLA Authors.

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

#include "gtest/gtest.h"

#include "zkx/service/custom_call_status_internal.h"
#include "zkx/service/custom_call_status_test_c_caller.h"

namespace zkx {

TEST(ZkxCustomCallStatusTest, DefaultIsSuccess) {
  ZkxCustomCallStatus status;

  ASSERT_EQ(CustomCallStatusGetMessage(&status), std::nullopt);
}

TEST(ZkxCustomCallStatusTest, SetSuccess) {
  ZkxCustomCallStatus status;
  ZkxCustomCallStatusSetSuccess(&status);

  ASSERT_EQ(CustomCallStatusGetMessage(&status), std::nullopt);
}

TEST(ZkxCustomCallStatusTest, SetSuccessAfterFailure) {
  ZkxCustomCallStatus status;
  ZkxCustomCallStatusSetFailure(&status, "error", 5);
  ZkxCustomCallStatusSetSuccess(&status);

  ASSERT_EQ(CustomCallStatusGetMessage(&status), std::nullopt);
}

TEST(ZkxCustomCallStatusTest, SetFailure) {
  ZkxCustomCallStatus status;
  ZkxCustomCallStatusSetFailure(&status, "error", 5);

  ASSERT_EQ(CustomCallStatusGetMessage(&status), "error");
}

TEST(ZkxCustomCallStatusTest, SetFailureAfterSuccess) {
  ZkxCustomCallStatus status;
  ZkxCustomCallStatusSetSuccess(&status);
  ZkxCustomCallStatusSetFailure(&status, "error", 5);

  ASSERT_EQ(CustomCallStatusGetMessage(&status), "error");
}

TEST(ZkxCustomCallStatusTest, SetFailureTruncatesErrorAtGivenLength) {
  ZkxCustomCallStatus status;
  ZkxCustomCallStatusSetFailure(&status, "error", 4);

  ASSERT_EQ(CustomCallStatusGetMessage(&status),
            "erro");  // codespell:ignore erro
}

TEST(ZkxCustomCallStatusTest, SetFailureTruncatesErrorAtNullTerminator) {
  ZkxCustomCallStatus status;
  ZkxCustomCallStatusSetFailure(&status, "error", 100);

  ASSERT_EQ(CustomCallStatusGetMessage(&status), "error");
}

// Test that the API works when called from pure C code.

TEST(ZkxCustomCallStatusTest, CSetSuccess) {
  ZkxCustomCallStatus status;
  CSetSuccess(&status);

  ASSERT_EQ(CustomCallStatusGetMessage(&status), std::nullopt);
}

TEST(ZkxCustomCallStatusTest, CSetFailure) {
  ZkxCustomCallStatus status;
  CSetFailure(&status, "error", 5);

  ASSERT_EQ(CustomCallStatusGetMessage(&status), "error");
}

}  // namespace zkx
