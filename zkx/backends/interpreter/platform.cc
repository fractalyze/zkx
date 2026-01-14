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

#include "zkx/backends/interpreter/platform.h"

#include <utility>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"

#include "zkx/backends/interpreter/executor.h"
#include "zkx/stream_executor/platform/initialize.h"
#include "zkx/stream_executor/platform_manager.h"

namespace stream_executor::interpreter {

ZkxInterpreterPlatform::ZkxInterpreterPlatform(const std::string& name,
                                               const Platform::Id& id)
    : name_(name), id_(id) {}

ZkxInterpreterPlatform::~ZkxInterpreterPlatform() {}

Platform::Id ZkxInterpreterPlatform::id() const { return id_; }

int ZkxInterpreterPlatform::VisibleDeviceCount() const { return 1; }

std::string_view ZkxInterpreterPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
ZkxInterpreterPlatform::DescriptionForDevice(int ordinal) const {
  return ZkxInterpreterExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> ZkxInterpreterPlatform::FindExisting(
    int ordinal) {
  return executor_cache_.Get(ordinal);
}

absl::StatusOr<StreamExecutor*> ZkxInterpreterPlatform::ExecutorForDevice(
    int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
ZkxInterpreterPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<ZkxInterpreterExecutor>(ordinal, this);
  auto init_status = executor->Init();
  if (!init_status.ok()) {
    return absl::Status{
        absl::StatusCode::kInternal,
        absl::StrFormat(
            "failed initializing StreamExecutor for device ordinal %d: %s",
            ordinal, init_status.ToString())};
  }

  return std::move(executor);
}

namespace {

void InitializeZkxInterpreterPlatform() {
  std::unique_ptr<Platform> platform(new ZkxInterpreterPlatform);
  CHECK_OK(PlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace
}  // namespace stream_executor::interpreter

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    interpreter_platform, se::interpreter::InitializeZkxInterpreterPlatform());
