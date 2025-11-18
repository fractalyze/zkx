/* Copyright 2018 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/service/hlo_proto_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zkx/service/hlo.pb.h"

namespace zkx {
namespace {

TEST(HloProtoUtilTest, ParamsAndOutputShapeMissingModule) {
  HloProto hlo_proto;

  auto status = EntryComputationParameterShapes(hlo_proto).status();
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.message(), ::testing::HasSubstr("missing HloModuleProto"));
}

TEST(HloProtoUtilTest, MissingProgramShape) {
  HloProto hlo_proto;
  HloModuleProto* module = hlo_proto.mutable_hlo_module();
  module->set_name("entry");

  auto status = EntryComputationParameterShapes(hlo_proto).status();
  ASSERT_FALSE(status.ok());
  ASSERT_THAT(status.message(), ::testing::HasSubstr("missing program shape"));
}

}  // namespace
}  // namespace zkx
