/* Copyright 2024 The OpenXLA Authors.

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

#include "zkx/backends/gpu/collectives/gpu_clique_key.h"

#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/casts.h"

namespace zkx::gpu {

CollectiveStreamId GetCollectiveStreamId(bool is_async,
                                         AsyncStreamKind stream_kind) {
  // TODO(ezhulenev): This implementation does not look correct as stream IDs
  // are not really unique. Figure out if it's the case and fix either the code
  // or the documentation.
  int64_t stream_id = static_cast<int64_t>(stream_kind);
  return CollectiveStreamId(is_async ? stream_id + 1 : 0);
}

GpuCliqueKey::GpuCliqueKey(
    std::vector<GlobalDeviceId> devices, CollectiveStreamId stream_id,
    AsyncStreamKind stream_kind,
    std::vector<std::vector<GlobalDeviceId>> participant_groups,
    GlobalDeviceId root_device)
    : CliqueKey(std::move(devices)),
      stream_id_(stream_id),
      stream_kind_(stream_kind),
      participant_groups_(std::move(participant_groups)),
      root_device_(root_device) {
  for (std::vector<GlobalDeviceId>& group : participant_groups_) {
    absl::c_sort(group);
  }
  // Compare the groups by their first element.
  auto compare_groups = [](const std::vector<GlobalDeviceId>& lhs,
                           const std::vector<GlobalDeviceId>& rhs) {
    CHECK(!lhs.empty());
    CHECK(!rhs.empty());
    return lhs[0] < rhs[0];
  };
  absl::c_sort(participant_groups_, compare_groups);
}

CollectiveStreamId GpuCliqueKey::stream_id() const { return stream_id_; }

GlobalDeviceId GpuCliqueKey::root_device() const { return root_device_; }

bool GpuCliqueKey::IsSubsetOf(const CliqueKey& other) const {
  auto* other_nccl = tsl::down_cast<const GpuCliqueKey*>(&other);
  if (other_nccl == nullptr) return false;

  return stream_id_ == other_nccl->stream_id_ &&
         absl::c_all_of(devices(), [&](GlobalDeviceId id) {
           return absl::c_linear_search(other_nccl->devices(), id);
         });
}

std::vector<GpuCliqueKey> GpuCliqueKey::GetSubKeys(int64_t nroots) const {
  const auto& devs = devices();
  int64_t nranks = devs.size();
  CHECK_LE(nroots, nranks);
  int64_t rank_per_root = nranks / nroots;
  int64_t rank_rem = nranks % nroots;
  std::vector<GpuCliqueKey> subkeys;
  for (int64_t i = 0; i < nroots; ++i) {
    GpuCliqueKey subkey(*this);
    if (i < rank_rem) {
      subkey.root_device_ = devs[i * (rank_per_root + 1)];
    } else {
      subkey.root_device_ =
          devs[rank_rem * (rank_per_root + 1) + (i - rank_rem) * rank_per_root];
    }
    subkeys.push_back(subkey);
  }
  return subkeys;
}

std::string GpuCliqueKey::ToString() const {
  std::string group_string = "";
  if (!participant_groups_.empty()) {
    std::vector<std::string> values;
    values.reserve(participant_groups_.size());
    for (const auto& group : participant_groups_) {
      values.push_back("[" + GlobalDeviceIdsToString(group) + "]");
    }
    group_string = absl::StrFormat("; groups=[%s]", absl::StrJoin(values, ","));
  }
  return absl::StrFormat("devices=[%s]; stream=%d%s; root_device=%lld",
                         GlobalDeviceIdsToString(devices()), stream_id_.value(),
                         group_string, root_device_.value());
}

void GpuCliqueKey::HashValue(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices(), stream_id_,
                           participant_groups_, root_device_);
}

bool operator==(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  return a.devices() == b.devices() && a.stream_id_ == b.stream_id_ &&
         a.participant_groups_ == b.participant_groups_ &&
         a.root_device_ == b.root_device_;
}

bool operator<(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  if (a.devices().size() < b.devices().size()) return true;
  if (b.devices().size() < a.devices().size()) return false;

  if (a.devices() < b.devices()) return true;
  if (b.devices() < a.devices()) return false;

  if (a.root_device_ < b.root_device_) return true;
  if (b.root_device_ < a.root_device_) return false;

  // NOTE(chokobole): Gemini suggested reversing this comparison, but doing so
  // caused the unittests to fail. To resolve this, the unittest was updated
  // to use the opposite comparison.
  // See: GpuCliqueKeyTest.Compare in gpu_clique_key_unittest.cc
  // Discussion: https://github.com/zk-rabbit/zkx/pull/31#discussion_r2266711030
  return a.stream_id_.value() > b.stream_id_.value();
}

bool operator>(const GpuCliqueKey& a, const GpuCliqueKey& b) {
  if (a.devices().size() > b.devices().size()) return true;
  if (b.devices().size() > a.devices().size()) return false;

  if (a.devices() > b.devices()) return true;
  if (b.devices() > a.devices()) return false;

  if (a.root_device_ > b.root_device_) return true;
  if (b.root_device_ > a.root_device_) return false;

  // To acquire sync cliques before async ones, we define an order where sync
  // cliques are 'greater'. Since sync stream IDs are smaller (0) than async
  // stream IDs (>0), we use `<` on the stream ID values to achieve this.
  return a.stream_id_.value() < b.stream_id_.value();
}

}  // namespace zkx::gpu
