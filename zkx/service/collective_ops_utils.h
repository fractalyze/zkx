/* Copyright 2019 The OpenXLA Authors.

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

#ifndef ZKX_SERVICE_COLLECTIVE_OPS_UTILS_H_
#define ZKX_SERVICE_COLLECTIVE_OPS_UTILS_H_

#include <stdint.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"

#include "zkx/executable_run_options.h"
#include "zkx/service/global_device_id.h"

namespace zkx {

enum class ReductionKind { kSum, kProduct, kMin, kMax };

std::string_view ReductionKindToString(ReductionKind reduction_kind);

// Key that identifies a particular Rendezvous object in our global hashtable.
// This determines which calls to ExecuteOnStream communicate with each other.
// The rules are as follows.
//
// * Only ops with the same RunId can communicate with each other. (This is the
//   whole purpose of RunId).
//
// * Only ops with the same set of participating replicas can communicate with
//   each other.  This is how we separate out different replica groups (e.g. a
//   single AllReduce HLO might do two reductions, between say GPUs {0,2} and
//   {1,3}).
//
// * Only ops with the same opcode can communicate with each other. At the
//   moment we only support kAllReduce, so we don't check for this explicitly.
//
// * For cross-module all-reduces (i.e. instr->channel_id().has_value()),
//   only ops with the same value for channel_id() can communicate with each
//   other.
//
// * For cross-replica (i.e. same-module) all-reduces (i.e.
//   !channel_id().has_value()), only ops from the same module (as
//   identified by its unique_id()) can communicate with each other.
//
struct RendezvousKey {
  enum CollectiveOpKind {
    kCrossModule,
    kCrossReplica,
  };

  explicit RendezvousKey(const RunId& run_id,
                         std::vector<GlobalDeviceId> global_devices,
                         int num_local_participants,
                         CollectiveOpKind collective_op_kind, int64_t op_id)
      : run_id(run_id),
        global_devices(std::move(global_devices)),
        num_local_participants(num_local_participants),
        collective_op_kind(collective_op_kind),
        op_id(op_id) {}

  template <typename H>
  friend H AbslHashValue(H h, const RendezvousKey& k) {
    return H::combine(std::move(h), k.run_id, k.global_devices,
                      k.num_local_participants, k.collective_op_kind, k.op_id);
  }
  friend bool operator==(const RendezvousKey& a, const RendezvousKey& b) {
    return a.run_id == b.run_id && a.global_devices == b.global_devices &&
           a.num_local_participants == b.num_local_participants &&
           a.collective_op_kind == b.collective_op_kind &&  //
           a.op_id == b.op_id;
  }
  friend bool operator!=(const RendezvousKey& a, const RendezvousKey& b) {
    return !(a == b);
  }

  std::string_view CollectiveOpKindString() const {
    switch (collective_op_kind) {
      case kCrossModule:
        return "cross_module";
      case kCrossReplica:
        return "cross_replica";
    }
  }

  std::string ToString() const {
    return absl::StrFormat(
        "RendezvousKey{run_id=%s, global_devices=[%s], "
        "num_local_participants=%d, collective_op_kind=%s, op_id=%d}",
        run_id.ToString(), GlobalDeviceIdsToString(global_devices),
        num_local_participants, CollectiveOpKindString(), op_id);
  }

  RunId run_id;
  std::vector<GlobalDeviceId> global_devices;
  int num_local_participants;
  CollectiveOpKind collective_op_kind;
  int64_t op_id;
};

}  // namespace zkx

#endif  // ZKX_SERVICE_COLLECTIVE_OPS_UTILS_H_
