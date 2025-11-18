/* Copyright 2020 The OpenXLA Authors.
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

#include "zkx/pjrt/distributed/topology_util.h"

#include "absl/status/status_matchers.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/thread_pool.h"
#include "zkx/pjrt/distributed/in_memory_key_value_store.h"

namespace zkx {

TEST(TopologyTest, BuildGlobalTopology) {
  std::vector<LocalTopologyProto> locals(2);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  GlobalTopologyProto global =
      BuildGlobalTopology(absl::Span<LocalTopologyProto>(locals),
                          /*assign_global_device_ids=*/true);
  EXPECT_EQ(global.nodes_size(), 2);
  EXPECT_EQ(global.nodes()[0].devices_size(), 2);
  EXPECT_EQ(global.nodes()[1].devices_size(), 2);
}

TEST(TopologyTest, ExchangeTopology) {
  int num_nodes = 2;
  std::vector<LocalTopologyProto> locals(num_nodes);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  InMemoryKeyValueStore kv_store;
  std::vector<GlobalTopologyProto> globals(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "TestPool",
                                        num_nodes);
    for (int i = 0; i < num_nodes; i++) {
      thread_pool.Schedule([&, i] {
        TF_ASSERT_OK(ExchangeTopologies(
            /*platform=*/"cuda", /*node_id=*/i, num_nodes,
            /*get_local_topology_timeout=*/
            absl::Seconds(10), /*get_global_topology_timeout=*/
            absl::Seconds(10), &kv_store, locals[i], &globals[i],
            /*assign_global_device_ids=*/true));
      });
    }
  }
  for (const GlobalTopologyProto& global : globals) {
    EXPECT_EQ(global.nodes_size(), 2);
    EXPECT_EQ(global.nodes()[0].devices_size(), 2);
    EXPECT_EQ(global.nodes()[1].devices_size(), 2);
  }
}

TEST(TopologyTest, ExchangeTopology_Twice_Succeeds) {
  int num_nodes = 2;
  std::vector<LocalTopologyProto> locals(num_nodes);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  InMemoryKeyValueStore kv_store(/*allow_overwrite=*/false);
  std::vector<GlobalTopologyProto> globals(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "TestPool",
                                        num_nodes);
    for (int i = 0; i < num_nodes; i++) {
      thread_pool.Schedule([&, i] {
        TF_ASSERT_OK(ExchangeTopologies(
            /*platform=*/"cuda", /*node_id=*/i, num_nodes,
            /*get_local_topology_timeout=*/
            absl::Seconds(10), /*get_global_topology_timeout=*/
            absl::Seconds(10), &kv_store, locals[i], &globals[i],
            /*assign_global_device_ids=*/true));
        // Simulate node 1 restarting and exchanging topologies again.
        if (i == 1) {
          TF_ASSERT_OK(ExchangeTopologies(
              /*platform=*/"cuda", /*node_id=*/i, num_nodes,
              /*get_local_topology_timeout=*/
              absl::Seconds(10), /*get_global_topology_timeout=*/
              absl::Seconds(10), &kv_store, locals[i], &globals[i],
              /*assign_global_device_ids=*/true));
        }
      });
    }
  }
  for (const GlobalTopologyProto& global : globals) {
    EXPECT_EQ(global.nodes_size(), 2);
    EXPECT_EQ(global.nodes()[0].devices_size(), 2);
    EXPECT_EQ(global.nodes()[1].devices_size(), 2);
  }
}

TEST(TopologyTest, ExchangeTopology_TwiceWithDifferentLocalTopology_Fails) {
  int num_nodes = 2;
  std::vector<LocalTopologyProto> locals(num_nodes);
  DeviceProto* d0 = locals[0].add_devices();
  d0->set_local_device_ordinal(0);
  DeviceProto* d1 = locals[0].add_devices();
  d1->set_local_device_ordinal(0);
  DeviceProto* d2 = locals[1].add_devices();
  d2->set_local_device_ordinal(0);
  DeviceProto* d3 = locals[1].add_devices();
  d3->set_local_device_ordinal(1);

  InMemoryKeyValueStore kv_store(/*allow_overwrite=*/false);
  std::vector<GlobalTopologyProto> globals(num_nodes);
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "TestPool",
                                        num_nodes);
    for (int i = 0; i < num_nodes; i++) {
      thread_pool.Schedule([&, i] {
        TF_ASSERT_OK(ExchangeTopologies(
            /*platform=*/"cuda", /*node_id=*/i, num_nodes,
            /*get_local_topology_timeout=*/
            absl::Seconds(10), /*get_global_topology_timeout=*/
            absl::Seconds(10), &kv_store, locals[i], &globals[i],
            /*assign_global_device_ids=*/true));
        // Simulate node 1 restarting with different devices.
        if (i == 1) {
          DeviceProto* d4 = locals[1].add_devices();
          d4->set_local_device_ordinal(2);
          // This should fail because the local topology is unexpectedly
          // different.
          EXPECT_THAT(ExchangeTopologies(
                          /*platform=*/"cuda", /*node_id=*/i, num_nodes,
                          /*get_local_topology_timeout=*/
                          absl::Seconds(10), /*get_global_topology_timeout=*/
                          absl::Seconds(10), &kv_store, locals[i], &globals[i],
                          /*assign_global_device_ids=*/true),
                      absl_testing::StatusIs(absl::StatusCode::kInternal));
        }
      });
    }
  }
}

}  // namespace zkx
