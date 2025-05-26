/* Copyright 2018 The OpenXLA Authors.

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

#include "zkx/client/executable_build_options.h"

#include "absl/strings/str_format.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/debug_options_flags.h"
#include "zkx/execution_options_util.h"
#include "zkx/layout_util.h"
#include "zkx/shape_util.h"

namespace zkx {

DebugOptions* ExecutableBuildOptions::mutable_debug_options() {
  if (!has_debug_options()) {
    debug_options_ = GetDebugOptionsFromFlags();
  }
  return &debug_options_.value();
}

std::string ExecutableBuildOptions::ToString() const {
  std::string result_layout = "nullopt";
  if (result_layout_set_) {
    result_layout = ShapeUtil::HumanStringWithLayout(result_layout_);
  }
  return absl::StrFormat(
      "ExecutableBuildOptions{device_ordinal=%d, result_layout=%s, "
      "num_replicas=%d}",
      device_ordinal_, result_layout, num_replicas_);
}

absl::StatusOr<ExecutableBuildOptionsProto> ExecutableBuildOptions::ToProto()
    const {
  ExecutableBuildOptionsProto output;
  output.set_device_ordinal(device_ordinal());
  if (result_layout()) {
    *output.mutable_result_layout() = result_layout()->ToProto();
  }
  if (has_comp_envs()) {
    *output.mutable_comp_envs() = comp_envs().ToProto();
  }
  if (has_debug_options()) {
    *output.mutable_debug_options() = debug_options();
  }
  if (layout_canonicalization_callback_) {
    return absl::InvalidArgumentError(
        "Cannot serialize "
        "ExecutableBuildOptions::layout_canonicalization_callback");
  }
  if (compile_thread_pool() != nullptr) {
    return absl::InvalidArgumentError(
        "Cannot serialize ExecutableBuildOptions::compile_thread_pool");
  }
  output.set_num_replicas(num_replicas());
  output.set_num_partitions(num_partitions());
  output.set_use_spmd_partitioning(use_spmd_partitioning());
  output.set_use_auto_spmd_partitioning(use_auto_spmd_partitioning());
  output.set_exec_time_optimization_effort(exec_time_optimization_effort());
  output.set_memory_fitting_effort(memory_fitting_effort());
  output.set_deduplicate_hlo(deduplicate_hlo());
  if (has_device_assignment()) {
    device_assignment().Serialize(output.mutable_device_assignment());
  }
  output.set_alias_passthrough_params(alias_passthrough_params());
  output.set_run_backend_only(run_backend_only());
  if (!allow_spmd_sharding_propagation_to_parameters().empty()) {
    output.mutable_allow_spmd_sharding_propagation_to_parameters()->Clear();
    for (bool v : allow_spmd_sharding_propagation_to_parameters()) {
      output.mutable_allow_spmd_sharding_propagation_to_parameters()->Add(v);
    }
  }
  if (!allow_spmd_sharding_propagation_to_output().empty()) {
    output.mutable_allow_spmd_sharding_propagation_to_output()->Clear();
    for (bool v : allow_spmd_sharding_propagation_to_output()) {
      output.mutable_allow_spmd_sharding_propagation_to_output()->Add(v);
    }
  }
  *output.mutable_fdo_profile() = fdo_profile();
  output.set_device_memory_size(device_memory_size());
  for (int64_t s : auto_spmd_partitioning_mesh_shape()) {
    output.mutable_auto_spmd_partitioning_mesh_shape()->Add(s);
  }
  for (int64_t s : auto_spmd_partitioning_mesh_ids()) {
    output.mutable_auto_spmd_partitioning_mesh_ids()->Add(s);
  }
  output.set_use_shardy_partitioner(use_shardy_partitioner());
  output.set_process_index(process_index());
  output.set_process_count(process_count());
  return output;
}

absl::StatusOr<ExecutableBuildOptions> ExecutableBuildOptionsFromProto(
    const ExecutableBuildOptionsProto& input) {
  ExecutableBuildOptions output;
  if (input.device_ordinal() != -1) {
    output.set_device_ordinal(input.device_ordinal());
  }
  if (input.has_result_layout()) {
    output.set_result_layout(Shape(input.result_layout()));
  }
  if (input.has_comp_envs()) {
    TF_ASSIGN_OR_RETURN(
        auto comp_envs,
        CompilationEnvironments::CreateFromProto(input.comp_envs()));
    *output.mutable_comp_envs() = std::move(*comp_envs);
  }
  if (input.has_debug_options()) {
    *output.mutable_debug_options() = input.debug_options();
  }
  output.set_num_replicas(input.num_replicas());
  output.set_num_partitions(input.num_partitions());
  output.set_use_spmd_partitioning(input.use_spmd_partitioning());
  output.set_use_auto_spmd_partitioning(input.use_auto_spmd_partitioning());
  output.set_exec_time_optimization_effort(
      input.exec_time_optimization_effort());
  output.set_memory_fitting_effort(input.memory_fitting_effort());
  output.set_deduplicate_hlo(input.deduplicate_hlo());
  if (input.has_device_assignment()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<DeviceAssignment> assignment,
        DeviceAssignment::Deserialize(input.device_assignment()));
    output.set_device_assignment(*assignment);
  }
  output.set_alias_passthrough_params(input.alias_passthrough_params());
  output.set_run_backend_only(input.run_backend_only());
  output.set_allow_spmd_sharding_propagation_to_parameters(
      input.allow_spmd_sharding_propagation_to_parameters());
  output.set_allow_spmd_sharding_propagation_to_output(
      input.allow_spmd_sharding_propagation_to_output());
  *output.mutable_fdo_profile() = input.fdo_profile();
  output.set_device_memory_size(input.device_memory_size());
  output.set_auto_spmd_partitioning_mesh_shape(
      std::vector<int64_t>(input.auto_spmd_partitioning_mesh_shape().begin(),
                           input.auto_spmd_partitioning_mesh_shape().end()));
  output.set_auto_spmd_partitioning_mesh_ids(
      std::vector<int64_t>(input.auto_spmd_partitioning_mesh_ids().begin(),
                           input.auto_spmd_partitioning_mesh_ids().end()));
  output.set_use_shardy_partitioner(input.use_shardy_partitioner());
  output.set_process_index(input.process_index());
  output.set_process_count(input.process_count());
  return output;
}

ExecutionOptions CreateExecutionOptions(
    const ExecutableBuildOptions& build_options,
    const ProgramShape* program_shape) {
  ExecutionOptions execution_options = CreateDefaultExecutionOptions();
  if (build_options.has_debug_options()) {
    *execution_options.mutable_debug_options() = build_options.debug_options();
  }
  if (build_options.result_layout() != nullptr) {
    *execution_options.mutable_shape_with_output_layout() =
        build_options.result_layout()->ToProto();
  } else {
    Shape result_shape(program_shape->result());
    LayoutUtil::SetToDefaultLayout(&result_shape);
    *execution_options.mutable_shape_with_output_layout() =
        result_shape.ToProto();
  }
  execution_options.set_num_replicas(build_options.num_replicas());
  execution_options.set_num_partitions(build_options.num_partitions());
  execution_options.set_use_spmd_partitioning(
      build_options.use_spmd_partitioning());
  execution_options.set_use_auto_spmd_partitioning(
      build_options.use_auto_spmd_partitioning());
  for (auto t : build_options.auto_spmd_partitioning_mesh_shape()) {
    execution_options.mutable_auto_spmd_partitioning_mesh_shape()->Add(t);
  }
  for (auto t : build_options.auto_spmd_partitioning_mesh_ids()) {
    execution_options.mutable_auto_spmd_partitioning_mesh_ids()->Add(t);
  }
  execution_options.set_exec_time_optimization_effort(
      build_options.exec_time_optimization_effort());
  execution_options.set_memory_fitting_effort(
      build_options.memory_fitting_effort());
  execution_options.set_deduplicate_hlo(build_options.deduplicate_hlo());
  if (!build_options.allow_spmd_sharding_propagation_to_parameters().empty()) {
    execution_options.mutable_allow_spmd_sharding_propagation_to_parameters()
        ->Clear();
    for (bool v :
         build_options.allow_spmd_sharding_propagation_to_parameters()) {
      execution_options.mutable_allow_spmd_sharding_propagation_to_parameters()
          ->Add(v);
    }
  }
  if (!build_options.allow_spmd_sharding_propagation_to_output().empty()) {
    execution_options.mutable_allow_spmd_sharding_propagation_to_output()
        ->Clear();
    for (bool v : build_options.allow_spmd_sharding_propagation_to_output()) {
      execution_options.mutable_allow_spmd_sharding_propagation_to_output()
          ->Add(v);
    }
  }
  if (build_options.has_device_assignment()) {
    build_options.device_assignment().Serialize(
        execution_options.mutable_device_assignment());
  }
  execution_options.set_alias_passthrough_params(
      build_options.alias_passthrough_params());
  execution_options.set_fdo_profile(build_options.fdo_profile().data(),
                                    build_options.fdo_profile().size());
  execution_options.set_device_memory_size(build_options.device_memory_size());
  execution_options.set_use_shardy_partitioner(
      build_options.use_shardy_partitioner());
  return execution_options;
}

}  // namespace zkx
