/* Copyright 2022 The OpenXLA Authors.
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

#include "zkx/python/ifrt/executable.h"

#include "xla/tsl/platform/statusor.h"

namespace zkx::ifrt {

char Executable::ID = 0;
char LoadedExecutable::ID = 0;
[[maybe_unused]] char ExecutableVersion::ID = 0;

absl::StatusOr<ExecuteOptionsProto> ExecuteOptions::ToProto(
    SerDesVersion version) const {
  if (version.version_number() != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version.version_number(),
                     " for ExecuteOptions serialization"));
  }
  ExecuteOptionsProto proto;
  proto.set_version_number(version.version_number().value());

  proto.set_launch_id(launch_id);
  proto.mutable_non_donatable_input_indices()->Add(
      non_donatable_input_indices.begin(), non_donatable_input_indices.end());
  proto.set_fill_status(fill_status);
  proto.set_execution_stream_id(execution_stream_id);
  if (custom_options.has_value()) {
    *proto.mutable_custom_options() = custom_options->ToProto(version);
  }

  return proto;
}

// static
absl::StatusOr<ExecuteOptions> ExecuteOptions::FromProto(
    const ExecuteOptionsProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for ExecuteOptions deserialization"));
  }

  ExecuteOptions options;
  options.launch_id = proto.launch_id();
  options.non_donatable_input_indices.insert(
      proto.non_donatable_input_indices().begin(),
      proto.non_donatable_input_indices().end());
  options.fill_status = proto.fill_status();
  options.execution_stream_id = proto.execution_stream_id();
  if (proto.has_custom_options()) {
    TF_ASSIGN_OR_RETURN(options.custom_options,
                        AttributeMap::FromProto(proto.custom_options()));
  }
  return options;
}

}  // namespace zkx::ifrt
