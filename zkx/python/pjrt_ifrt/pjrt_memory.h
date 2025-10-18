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

#ifndef ZKX_PYTHON_PJRT_IFRT_PJRT_MEMORY_H_
#define ZKX_PYTHON_PJRT_IFRT_PJRT_MEMORY_H_

#include <vector>

#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include "zkx/pjrt/pjrt_client.h"
#include "zkx/python/ifrt/memory.h"

namespace zkx::ifrt {

class PjRtClient;

class PjRtCompatibleMemory
    : public llvm::RTTIExtends<PjRtCompatibleMemory, Memory> {
 public:
  virtual zkx::PjRtMemorySpace* pjrt_memory() = 0;

  static char ID;
};

class PjRtMemory final
    : public llvm::RTTIExtends<PjRtMemory, PjRtCompatibleMemory> {
 public:
  PjRtMemory(PjRtClient* client, zkx::PjRtMemorySpace* pjrt_memory);

  // Constructor for memories for non-addressable devices that are not backed by
  // a PjRtMemorySpace.
  PjRtMemory(PjRtClient* client, const MemoryKind& kind, Device* device);

  PjRtClient* client() const { return client_; }
  zkx::PjRtMemorySpace* pjrt_memory() override { return pjrt_memory_; }

  MemoryId Id() const override;
  const MemoryKind& Kind() const override;
  std::string_view ToString() const override;
  std::string_view DebugString() const override;
  absl::Span<Device* const> Devices() const override;

  static char ID;

 private:
  PjRtClient* client_;                 // not owned
  zkx::PjRtMemorySpace* pjrt_memory_;  // not owned
  MemoryKind kind_;
  std::vector<Device*> devices_;
};

// Canonicalizes `MemoryKind`. If `MemoryKind` has no memory kind chosen,
// returns a default `MemoryKind` chosen for the PjRt device. If there is no
// default indicated by the device, simply returns `MemoryKind` with no memory
// kind chosen.
//
// TODO(hyeontaek,yashkatariya): Harden `MemoryKind` creation paths so that
// every `MemoryKind` is canonicalized and does not require on-demand
// canonicalization.
MemoryKind CanonicalizeMemoryKindWithPjRtDevice(MemoryKind memory_kind,
                                                zkx::PjRtDevice* device);

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_PJRT_IFRT_PJRT_MEMORY_H_
