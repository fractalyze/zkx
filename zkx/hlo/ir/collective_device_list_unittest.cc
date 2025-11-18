/* Copyright 2024 The OpenXLA Authors.
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

#include "zkx/hlo/ir/collective_device_list.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace zkx {

TEST(CollectiveDeviceListTest, DefaultListToString) {
  EXPECT_EQ(CollectiveDeviceList().ToString(), "{}");
  EXPECT_EQ(CollectiveDeviceList({{1, 2}, {3, 4}}).ToString(), "{{1,2},{3,4}}");
  EXPECT_EQ(CollectiveDeviceList({{1, 2, 3, 4, 5, 6, 7}}).ToString(),
            "{{1,2,3,4,5,6,7}}");
}

TEST(CollectiveDeviceListTest, DeepCopy) {
  CollectiveDeviceList orig({{1, 2, 3, 4, 5, 6, 7}});
  CollectiveDeviceList copy = orig;
  EXPECT_EQ(&orig.replica_groups(), &copy.replica_groups());
}

TEST(CollectiveDeviceListTest, DefaultListToProto) {
  EXPECT_THAT(CollectiveDeviceList().ToProto().replica_groups().size(), 0);
  CollectiveDeviceList list({{1, 2}, {3, 4}});
  CollectiveDeviceListProto proto = list.ToProto();
  EXPECT_THAT(proto.replica_groups().size(), 2);
  EXPECT_THAT(proto.replica_groups(0).replica_ids(),
              testing::ElementsAre(1, 2));
  EXPECT_THAT(proto.replica_groups(1).replica_ids(),
              testing::ElementsAre(3, 4));
  EXPECT_FALSE(proto.has_iota_replica_group_list());
}

TEST(CollectiveDeviceListTest, DefaultListToProto2) {
  CollectiveDeviceList list({{1, 2, 3, 4, 5, 6, 7}});
  CollectiveDeviceListProto proto = list.ToProto();
  EXPECT_THAT(proto.replica_groups().size(), 1);
  EXPECT_THAT(proto.replica_groups(0).replica_ids(),
              testing::ElementsAre(1, 2, 3, 4, 5, 6, 7));
  EXPECT_FALSE(proto.has_iota_replica_group_list());
}

TEST(CollectiveDeviceListTest, IotaListToString) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10));
  EXPECT_EQ(list.ToString(), "[2,10]<=[20]");
}

TEST(CollectiveDeviceListTest,
     IotaListToStringWithPrintingFullReplicaGroupList) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10));
  EXPECT_EQ(list.ToString(/*print_full_replica_group_list=*/true),
            "{{0,1,2,3,4,5,6,7,8,9},{10,11,12,13,14,15,16,17,18,19}}");
}

TEST(CollectiveDeviceListTest, IotaListToString2) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10, {4, 5}, {1, 0}));
  EXPECT_EQ(list.ToString(), "[2,10]<=[4,5]T(1,0)");
}

TEST(CollectiveDeviceListTest,
     IotaListToStringWithPrintingFullReplicaGroupList2) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10, {4, 5}, {1, 0}));
  EXPECT_EQ(list.ToString(/*print_full_replica_group_list=*/true),
            "{{0,5,10,15,1,6,11,16,2,7},{12,17,3,8,13,18,4,9,14,19}}");
}

TEST(CollectiveDeviceListTest, IotaListToProto) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10));
  CollectiveDeviceListProto proto = list.ToProto();
  EXPECT_EQ(proto.iota_replica_group_list().num_replica_groups(), 2);
  EXPECT_EQ(proto.iota_replica_group_list().num_devices_per_group(), 10);
  EXPECT_THAT(proto.iota_replica_group_list().iota_reshape_dims(),
              testing::ElementsAre(20));
  EXPECT_THAT(proto.iota_replica_group_list().iota_transpose_perm(),
              testing::ElementsAre(0));
  EXPECT_THAT(proto.replica_groups_size(), 0);
}

TEST(CollectiveDeviceListTest, IotaListToProto2) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10, {4, 5}, {1, 0}));
  CollectiveDeviceListProto proto = list.ToProto();
  EXPECT_EQ(proto.iota_replica_group_list().num_replica_groups(), 2);
  EXPECT_EQ(proto.iota_replica_group_list().num_devices_per_group(), 10);
  EXPECT_THAT(proto.iota_replica_group_list().iota_reshape_dims(),
              testing::ElementsAre(4, 5));
  EXPECT_THAT(proto.iota_replica_group_list().iota_transpose_perm(),
              testing::ElementsAre(1, 0));
  EXPECT_THAT(proto.replica_groups_size(), 0);
}

}  // namespace zkx
