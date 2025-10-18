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

#ifndef ZKX_PYTHON_PJRT_IFRT_PJRT_DEVICE_H_
#define ZKX_PYTHON_PJRT_IFRT_PJRT_DEVICE_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"

#include "zkx/pjrt/pjrt_client.h"
#include "zkx/pjrt/pjrt_device_description.h"
#include "zkx/python/ifrt/attribute_map.h"
#include "zkx/python/ifrt/device.h"
#include "zkx/python/pjrt_ifrt/pjrt_client.h"

namespace zkx::ifrt {

class PjRtCompatibleDevice
    : public llvm::RTTIExtends<PjRtCompatibleDevice, Device> {
 public:
  virtual zkx::PjRtDevice* pjrt_device() const = 0;

  static char ID;
};

class PjRtDevice final
    : public llvm::RTTIExtends<PjRtDevice, PjRtCompatibleDevice> {
 public:
  PjRtDevice(PjRtClient* client, DeviceId id, std::string kind,
             std::string to_string, std::string debug_string, int process_index,
             absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes,
             zkx::PjRtDevice* pjrt_device);

  // Non-null only for addressable devices. nullptr for non-addressable devices.
  zkx::PjRtDevice* pjrt_device() const override { return pjrt_device_; }

  // Device implementation.

  PjRtClient* client() const override { return client_; }

  DeviceId Id() const final;
  const AttributeMap& Attributes() const final;
  std::string_view Kind() const final;
  std::string_view ToString() const final;
  std::string_view DebugString() const final;
  bool IsAddressable() const final;
  absl::StatusOr<Memory*> DefaultMemory() const final;
  absl::Span<Memory* const> Memories() const final;
  int ProcessIndex() const final;

  static char ID;

 private:
  friend class PjRtClient;

  PjRtClient* client_;  // not owned

  DeviceId id_;
  AttributeMap attributes_;
  std::string kind_;
  std::string to_string_;
  std::string debug_string_;
  absl::StatusOr<Memory*> default_memory_;
  std::vector<Memory*> memories_;
  int process_index_;

  zkx::PjRtDevice* pjrt_device_;  // not owned
};

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_PJRT_IFRT_PJRT_DEVICE_H_
