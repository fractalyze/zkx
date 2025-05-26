/* Copyright 2020 The OpenXLA Authors.

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

#include "zkx/pjrt/utils.h"

#include <algorithm>
#include <cstdlib>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"

#include "xla/tsl/platform/cpu_info.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/base/logging.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/ir/hlo_sharding.h"
#include "zkx/layout_util.h"
#include "zkx/primitive_util.h"
#include "zkx/shape_util.h"

namespace zkx {
namespace {

absl::StatusOr<Shape> GetShardedShape(const Shape& shape,
                                      const OpSharding& sharding) {
  TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                      HloSharding::FromProto(sharding));
  if (shape.IsTuple()) {
    Shape sharded_shape = shape;
    ShapeUtil::ForEachMutableSubshape(
        &sharded_shape, [&](Shape* subshape, const ShapeIndex& index) {
          if (!subshape->IsTuple()) {
            HloSharding subsharding = hlo_sharding.GetSubSharding(shape, index);
            *subshape = subsharding.TileShape(*subshape);
          }
        });
    return sharded_shape;
  } else {
    return hlo_sharding.TileShape(shape);
  }
}

absl::StatusOr<Shape> GetShardedShape(const HloInstructionProto& instr) {
  const Shape unsharded_shape(instr.shape());
  Shape sharded_shape;
  if (instr.has_sharding()) {
    TF_ASSIGN_OR_RETURN(sharded_shape,
                        GetShardedShape(unsharded_shape, instr.sharding()));
  } else {
    sharded_shape = unsharded_shape;
  }
  LayoutUtil::ClearLayout(&sharded_shape);
  return sharded_shape;
}

// Returns sharded (argument shapes, result shape) without layouts.
absl::StatusOr<std::pair<std::vector<Shape>, Shape>> GetShardedProgramShapes(
    const ZkxComputation& computation, const ProgramShape& program_shape) {
  std::vector<Shape> arg_shapes;
  arg_shapes.resize(program_shape.parameters_size());
  Shape result_shape;
  for (const HloComputationProto& comp : computation.proto().computations()) {
    if (comp.id() != computation.proto().entry_computation_id()) {
      continue;
    }
    for (const HloInstructionProto& instr : comp.instructions()) {
      if (instr.opcode() == HloOpcodeString(HloOpcode::kParameter)) {
        if (instr.parameter_number() >= program_shape.parameters_size()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Got invalid parameter number %d, expected %d parameters",
              instr.parameter_number(), program_shape.parameters_size()));
        }
        TF_ASSIGN_OR_RETURN(arg_shapes[instr.parameter_number()],
                            GetShardedShape(instr));
      }
      if (instr.id() == comp.root_id()) {
        if (result_shape.element_type() != PRIMITIVE_TYPE_INVALID) {
          return absl::InvalidArgumentError("Found multiple root instructions");
        }
        TF_ASSIGN_OR_RETURN(result_shape, GetShardedShape(instr));
      }
    }
  }
  for (int i = 0; i < arg_shapes.size(); ++i) {
    if (arg_shapes[i].element_type() == PRIMITIVE_TYPE_INVALID) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Couldn't find parameter %d", i));
    }
  }
  if (result_shape.element_type() == PRIMITIVE_TYPE_INVALID) {
    return absl::InvalidArgumentError("Couldn't find root instruction");
  }
  return std::make_pair(arg_shapes, result_shape);
}

}  // namespace

absl::Status ParseDeviceAssignmentCompileOptions(
    bool compile_portable_executable, ExecutableBuildOptions* build_options,
    std::function<absl::StatusOr<DeviceAssignment>(int, int)>
        GetDefaultDeviceAssignmentFunction,
    int* num_replicas, int* num_partitions,
    std::shared_ptr<DeviceAssignment>* device_assignment) {
  if (compile_portable_executable) {
    if (build_options->has_device_assignment()) {
      return absl::InvalidArgumentError(
          "CompileOptions requests portable executable but "
          "ExecutableBuildOptions includes a device assignment");
    }
    if (build_options->num_replicas() != 1 ||
        build_options->num_partitions() != 1) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "CompileOptions requests portable executable but "
          "ExecutableBuildOptions includes num_replicas %d and num_partitions "
          "%d.",
          build_options->num_replicas(), build_options->num_partitions()));
    }
    *num_replicas = 1;
    *num_partitions = 1;
  } else {
    if (!build_options->has_device_assignment()) {
      VLOG(2) << "Compile using default device_assignment.";
      TF_ASSIGN_OR_RETURN(
          DeviceAssignment device_assignment,
          GetDefaultDeviceAssignmentFunction(build_options->num_replicas(),
                                             build_options->num_partitions()));
      build_options->set_device_assignment(device_assignment);
    }
    VLOG(2) << "Compile device_assignment:\n"
            << build_options->device_assignment().ToString();
    *num_replicas = build_options->device_assignment().replica_count();
    *num_partitions = build_options->device_assignment().computation_count();
    *device_assignment =
        std::make_shared<DeviceAssignment>(build_options->device_assignment());
  }
  return absl::OkStatus();
}

absl::Status DetermineArgumentLayoutsFromCompileOptions(
    const ZkxComputation& computation,
    std::function<absl::StatusOr<Shape>(Shape)>
        choose_compact_layout_for_shape_function,
    std::optional<std::vector<Shape>>& argument_layouts,
    ExecutableBuildOptions* build_options,
    std::vector<const Shape*>* argument_layout_pointers) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  const bool given_argument_layouts = argument_layouts.has_value();
  if (!argument_layouts) {
    argument_layouts.emplace(program_shape.parameters());
    for (Shape& shape : *argument_layouts) {
      LayoutUtil::ClearLayout(&shape);
    }
  } else if (argument_layouts->size() != program_shape.parameters_size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "CompileOptions specify %d argument layouts, but computation has %d "
        "arguments",
        argument_layouts->size(), program_shape.parameters_size()));
  }
  argument_layout_pointers->reserve(argument_layouts->size());

  // Assign a default layout based on `sharded_shape` to any array subshapes in
  // `dst_shape` that are missing layouts.
  auto assign_layouts = [&](const Shape& sharded_shape, Shape* dst_shape) {
    return ShapeUtil::ForEachMutableSubshapeWithStatus(
        dst_shape, [&](Shape* subshape, const ShapeIndex& idx) {
          if (subshape->IsArray() && !subshape->has_layout()) {
            CHECK(ShapeUtil::IndexIsValid(sharded_shape, idx));
            const Shape& sharded_subshape =
                ShapeUtil::GetSubshape(sharded_shape, idx);
            LayoutUtil::SetToDefaultLayout(subshape);
            TF_ASSIGN_OR_RETURN(
                Shape layout,
                choose_compact_layout_for_shape_function(sharded_subshape));
            if (layout.has_layout()) {
              *subshape->mutable_layout() = layout.layout();
            } else {
              subshape->clear_layout();
            }
          }
          return absl::OkStatus();
        });
  };
  TF_ASSIGN_OR_RETURN(auto sharded_shapes,
                      GetShardedProgramShapes(computation, program_shape));

  CHECK_EQ(sharded_shapes.first.size(), argument_layouts->size());
  for (int i = 0; i < argument_layouts->size(); ++i) {
    Shape* layout = &(*argument_layouts)[i];
    argument_layout_pointers->push_back(layout);
    TF_RETURN_IF_ERROR(assign_layouts(sharded_shapes.first[i], layout));
    if (!given_argument_layouts) {
      // Carry memory space forward from program shape.
      ShapeUtil::ForEachSubshape(
          program_shape.parameters(i),
          [&](const Shape& subshape, const ShapeIndex& index) {
            if (subshape.IsArray()) {
              Shape* program_subshape =
                  ShapeUtil::GetMutableSubshape(layout, index);
              if (program_subshape->has_layout() && subshape.has_layout()) {
                program_subshape->mutable_layout()->set_memory_space(
                    subshape.layout().memory_space());
              }
            }
          });
    }
  }

  bool used_program_shape;
  Shape result_layout;
  if (build_options->result_layout()) {
    result_layout = *build_options->result_layout();
    used_program_shape = false;
  } else {
    result_layout = program_shape.result();
    LayoutUtil::ClearLayout(&result_layout);
    used_program_shape = true;
  }
  TF_RETURN_IF_ERROR(assign_layouts(sharded_shapes.second, &result_layout));
  if (used_program_shape) {
    // Carry memory spaces forward from program shape.
    ShapeUtil::ForEachSubshape(
        program_shape.result(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsArray()) {
            Shape* result_subshape =
                ShapeUtil::GetMutableSubshape(&result_layout, index);
            if (result_subshape->has_layout() && subshape.has_layout()) {
              result_subshape->mutable_layout()->set_memory_space(
                  subshape.layout().memory_space());
            }
          }
        });
  }
  build_options->set_result_layout(result_layout);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> ComputeParametersThatMustBeDonated(
    const HloModule& module, bool tuple_inputs) {
  HloComputation* computation = module.entry_computation();
  int number_of_parameters = [&]() -> int {
    if (tuple_inputs) {
      CHECK_EQ(computation->num_parameters(), 1);
      const Shape& input_tuple_shape =
          computation->parameter_instruction(0)->shape();
      CHECK(input_tuple_shape.IsTuple());
      return input_tuple_shape.tuple_shapes_size();
    } else {
      return computation->num_parameters();
    }
  }();
  // If any buffer in a parameter is aliased we will donate the entire input
  // parameter.
  std::vector<int> parameters_to_donate;
  parameters_to_donate.reserve(computation->num_parameters());
  const HloInputOutputAliasConfig& config = module.input_output_alias_config();
  TF_RETURN_IF_ERROR(config.ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) -> absl::Status {
        if (tuple_inputs) {
          if (alias.parameter_number != 0) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Unexpected parameter number %d in alias config with tupled "
                "inputs",
                alias.parameter_number));
          }
          const ShapeIndex& index = alias.parameter_index;
          if (!index.empty()) {
            int this_parameter = index.data()[0];
            if (this_parameter >= number_of_parameters) {
              return absl::InvalidArgumentError(absl::StrFormat(
                  "Unexpected parameter index %s in alias config with tupled "
                  "inputs and %d parameters",
                  index.ToString(), number_of_parameters));
            }
            parameters_to_donate.push_back(this_parameter);
          }
        } else {
          int this_parameter = alias.parameter_number;
          if (this_parameter >= number_of_parameters) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Unexpected parameter number %d in alias config without tupled "
                "inputs and %d parameters",
                this_parameter, number_of_parameters));
          }
          parameters_to_donate.push_back(this_parameter);
        }
        return absl::OkStatus();
      }));
  absl::c_sort(parameters_to_donate);
  return parameters_to_donate;
}

int DefaultThreadPoolSize() {
  // Google's CI system exposes an environment variable NPROC that describes
  // a CPU reservation for tests.
  // TODO(phawkins): expose a better thought-out set of knobs to control
  // parallelism.
  for (const char* nproc_env : {"PJRT_NPROC", "NPROC"}) {
    const char* nproc_str = std::getenv(nproc_env);
    int nproc = 0;
    if (nproc_str && absl::SimpleAtoi(nproc_str, &nproc)) {
      return std::max(0, nproc);
    }
  }
  return tsl::port::MaxParallelism();
}

bool HasMajorToMinorLayout(PrimitiveType type, absl::Span<const int64_t> dims,
                           absl::Span<const int64_t> byte_strides) {
  CHECK_EQ(dims.size(), byte_strides.size());
  // If the array is size 0, the strides are irrelevant.
  if (absl::c_find(dims, 0) != dims.end()) {
    return true;
  }
  int64_t stride = primitive_util::ByteWidth(type);
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    // If a dimension is of size 1, its stride is irrelevant.
    if (dims[i] != 1) {
      if (byte_strides[i] != stride) {
        return false;
      }
      stride *= dims[i];
    }
  }
  return true;
}

absl::Status TestBufferDonationClashes(
    void* opaque_key,
    absl::flat_hash_map<const void*, std::pair<bool, int>>& donation_clashes,
    bool is_donated, int arg_idx, int replica, int partition) {
  auto [donation_clash_it, first_use] =
      donation_clashes.emplace(opaque_key, std::make_pair(is_donated, arg_idx));
  if (!first_use && (is_donated || donation_clash_it->second.first)) {
    auto [prev_is_donated, prev_arg_idx] = donation_clash_it->second;
    if (is_donated && prev_is_donated) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Attempt to donate the same buffer twice in Execute() ("
          "flattened argument %d, replica %d, partition %d, first use: %d). "
          "Toy "
          "example for this bug: `f(donate(a), donate(a))`.",
          arg_idx, replica, partition, prev_arg_idx));
    } else if (is_donated) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Attempt to donate a buffer which is also used by the same call "
          "to Execute() (flattened argument %d, replica %d, partition %d, "
          "first use: %d). Toy example for this bug: `f(a, donate(a))`.",
          arg_idx, replica, partition, prev_arg_idx));
    } else {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Attempt to use a buffer that was previously donated in the same "
          "call to Execute() (flattened argument %d, replica %d, partition "
          "%d, first use: %d). Toy example for this bug: `f(donate(a), "
          "a)`.",
          arg_idx, replica, partition, prev_arg_idx));
    }
  }
  return absl::OkStatus();
}

}  // namespace zkx
