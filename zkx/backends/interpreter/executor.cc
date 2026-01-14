/* Copyright 2017 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#include "zkx/backends/interpreter/executor.h"

#include <string.h>

#include "absl/log/check.h"

namespace stream_executor::interpreter {

host::HostStream *AsExecutorStream(Stream *stream) {
  DCHECK_NE(stream, nullptr);
  return dynamic_cast<host::HostStream *>(stream);
}

DeviceMemoryBase ZkxInterpreterExecutor::Allocate(uint64_t size,
                                                  int64_t memory_space) {
  return DeviceMemoryBase(new char[size], size);
}

void ZkxInterpreterExecutor::Deallocate(DeviceMemoryBase *mem) {
  delete[] static_cast<char *>(mem->opaque());
}

absl::Status ZkxInterpreterExecutor::SynchronousMemcpy(
    DeviceMemoryBase *dev_dst, const void *host_src, uint64_t size) {
  memcpy(dev_dst->opaque(), host_src, size);
  return absl::OkStatus();
}

absl::Status ZkxInterpreterExecutor::SynchronousMemcpy(
    void *host_dst, const DeviceMemoryBase &dev_src, uint64_t size) {
  memcpy(host_dst, dev_src.opaque(), size);
  return absl::OkStatus();
}

// static
absl::StatusOr<std::unique_ptr<DeviceDescription>>
ZkxInterpreterExecutor::CreateDeviceDescription(int device_ordinal) {
  DeviceDescription desc;

  desc.set_device_address_bits(64);

  desc.set_name("Interpreter");
  desc.set_device_memory_size(static_cast<uint64_t>(4) * 1024 * 1024 * 1024);
  desc.set_clock_rate_ghz(static_cast<float>(CLOCKS_PER_SEC) / 1e9);

  return std::make_unique<DeviceDescription>(std::move(desc));
}

}  // namespace stream_executor::interpreter
