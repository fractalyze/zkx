/* Copyright 2017 The OpenXLA Authors.

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

#include "zkx/debug_options_flags.h"

#include "absl/base/call_once.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "zkx/debug_options_parsers.h"
#include "zkx/parse_flags_from_env.h"

namespace zkx {

DebugOptions DefaultDebugOptionsIgnoringFlags() {
  DebugOptions opts;
  opts.set_zkx_llvm_enable_invariant_load_metadata(true);
  opts.set_zkx_llvm_disable_expensive_passes(false);
  opts.set_zkx_backend_optimization_level(3);

  opts.set_zkx_annotate_with_emitter_loc(false);
  opts.set_zkx_cpu_parallel_codegen_split_count(32);
  opts.set_zkx_cpu_enable_concurrency_optimized_scheduler(true);
  opts.set_zkx_cpu_prefer_vector_width(256);
  opts.set_zkx_cpu_max_isa("");

  opts.set_zkx_force_host_platform_device_count(1);

  opts.set_zkx_multiheap_size_constraint_per_heap(-1);

  opts.set_zkx_pjrt_allow_auto_layout_in_hlo(false);
  return opts;
}

static absl::once_flag flags_init;
static DebugOptions* flag_values;
static std::vector<tsl::Flag>* flag_objects;

void MakeDebugOptionsFlags(std::vector<tsl::Flag>* flag_list,
                           DebugOptions* debug_options) {
  // Returns a lambda that calls "member_setter" on "debug_options" with the
  // argument passed in to the lambda.
  auto bool_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(bool)) {
        return [debug_options, member_setter](bool value) {
          (debug_options->*member_setter)(value);
          return true;
        };
      };

  // Returns a lambda that calls "member_setter" on "debug_options" with the
  // argument passed in to the lambda.
  auto int32_setter_for =
      [debug_options](void (DebugOptions::*member_setter)(int32_t)) {
        return [debug_options, member_setter](int32_t value) {
          (debug_options->*member_setter)(value);
          return true;
        };
      };

  auto uppercase_string_setter_for =
      [debug_options](
          void (DebugOptions::*member_setter)(const std::string& value)) {
        return [debug_options, member_setter](const std::string& value) {
          (debug_options->*member_setter)(absl::AsciiStrToUpper(value));
          return true;
        };
      };

  // Custom "sub-parser" lambda for zkx_backend_extra_options.
  auto setter_for_zkx_backend_extra_options =
      [debug_options](std::string comma_separated_values) {
        auto* extra_options_map =
            debug_options->mutable_zkx_backend_extra_options();
        parse_zkx_backend_extra_options(extra_options_map,
                                        comma_separated_values);
        return true;
      };

  // Don't use an initializer list for initializing the vector; this would
  // create a temporary copy, and exceeds the stack space when compiling with
  // certain configurations.
  flag_list->push_back(tsl::Flag(
      "zkx_llvm_enable_invariant_load_metadata",
      bool_setter_for(
          &DebugOptions::set_zkx_llvm_enable_invariant_load_metadata),
      debug_options->zkx_llvm_enable_invariant_load_metadata(),
      "In LLVM-based backends, enable the emission of !invariant.load metadata "
      "in the generated IR."));
  flag_list->push_back(tsl::Flag(
      "zkx_llvm_disable_expensive_passes",
      bool_setter_for(&DebugOptions::set_zkx_llvm_disable_expensive_passes),
      debug_options->zkx_llvm_disable_expensive_passes(),
      "In LLVM-based backends, disable a custom set of expensive optimization "
      "passes."));
  flag_list->push_back(tsl::Flag(
      "zkx_backend_optimization_level",
      int32_setter_for(&DebugOptions::set_zkx_backend_optimization_level),
      debug_options->zkx_backend_optimization_level(),
      "Numerical optimization level for the ZKX compiler backend."));
  flag_list->push_back(tsl::Flag(
      "zkx_backend_extra_options", setter_for_zkx_backend_extra_options, "",
      "Extra options to pass to a backend; comma-separated list of 'key=val' "
      "strings (=val may be omitted); no whitespace around commas."));
  flag_list->push_back(tsl::Flag(
      "zkx_cpu_parallel_codegen_split_count",
      int32_setter_for(&DebugOptions::set_zkx_cpu_parallel_codegen_split_count),
      debug_options->zkx_cpu_parallel_codegen_split_count(),
      "Split LLVM module into at most this many parts before codegen to enable "
      "parallel compilation for the CPU backend."));
  flag_list->push_back(tsl::Flag(
      "zkx_cpu_enable_concurrency_optimized_scheduler",
      bool_setter_for(
          &DebugOptions::set_zkx_cpu_enable_concurrency_optimized_scheduler),
      debug_options->zkx_cpu_enable_concurrency_optimized_scheduler(),
      "Use HLO module scheduler that is optimized for extracting concurrency "
      "from an HLO module by trading off extra memory pressure."));
  flag_list->push_back(tsl::Flag(
      "zkx_cpu_prefer_vector_width",
      int32_setter_for(&DebugOptions::set_zkx_cpu_prefer_vector_width),
      debug_options->zkx_cpu_prefer_vector_width(),
      "Preferred vector width for the ZKX:CPU LLVM backend."));
  flag_list->push_back(tsl::Flag(
      "zkx_cpu_max_isa",
      uppercase_string_setter_for(&DebugOptions::set_zkx_cpu_max_isa),
      debug_options->zkx_cpu_max_isa(),
      "Maximum ISA that ZKX:CPU LLVM backend will codegen, i.e., it will not "
      "use newer instructions. Available values: SSE4_2, AVX, AVX2, AVX512, "
      "AVX512_VNNI, AVX512_BF16, AMX, and AMX_FP16. (`AMX` will enable both "
      "`AMX_BF16` and `AMX_INT8` instructions.)"));
  flag_list->push_back(tsl::Flag(
      "zkx_force_host_platform_device_count",
      int32_setter_for(&DebugOptions::set_zkx_force_host_platform_device_count),
      debug_options->zkx_force_host_platform_device_count(),
      "Force the host platform to pretend that there are these many host "
      "\"devices\". All of these host devices are backed by the same "
      "threadpool. Setting this to anything other than 1 can increase overhead "
      "from context switching but we let the user override this behavior to "
      "help run tests on the host that run models in parallel across multiple "
      "devices."));
  flag_list->push_back(tsl::Flag(
      "zkx_flags_reset", bool_setter_for(&DebugOptions::set_zkx_flags_reset),
      debug_options->zkx_flags_reset(),
      "Whether to reset XLA_FLAGS next time to parse."));
  flag_list->push_back(tsl::Flag(
      "zkx_annotate_with_emitter_loc",
      bool_setter_for(&DebugOptions::set_zkx_annotate_with_emitter_loc),
      debug_options->zkx_annotate_with_emitter_loc(),
      "Forces emitters that use MLIR to annotate all the created MLIR "
      "instructions with the emitter's C++ source file and line number. The "
      "annotations should appear in the MLIR dumps. The emitters should use "
      "EmitterLocOpBuilder for that."));
  flag_list->push_back(tsl::Flag(
      "zkx_dump_hlo_snapshots",
      bool_setter_for(&DebugOptions::set_zkx_dump_hlo_snapshots),
      debug_options->zkx_dump_hlo_snapshots(),
      "Every time an HLO module is run, dumps an HloSnapshot to the directory "
      "specified by --zkx_dump_to."));
  flag_list->push_back(tsl::Flag(
      "zkx_multiheap_size_constraint_per_heap",
      int32_setter_for(
          &DebugOptions::set_zkx_multiheap_size_constraint_per_heap),
      debug_options->zkx_multiheap_size_constraint_per_heap(),
      "Generates multiple heaps (i.e., temp buffers) with a size "
      "constraint on each heap to avoid Out-of-Memory due to memory "
      "fragmentation. The constraint is soft, so it works with tensors "
      "larger than the given constraint size. -1 corresponds to no "
      "constraints."));
  flag_list->push_back(tsl::Flag(
      "zkx_pjrt_allow_auto_layout_in_hlo",
      bool_setter_for(&DebugOptions::set_zkx_pjrt_allow_auto_layout_in_hlo),
      debug_options->zkx_pjrt_allow_auto_layout_in_hlo(),
      "Experimental: Make unset entry computation layout mean auto layout "
      "instead of default layout in HLO when run through PjRT. In other cases "
      "(StableHLO or non-PjRT) the auto layout is already used."));
}

// Allocates flag_values and flag_objects; this function must not be called more
// than once - its call done via call_once.
static void AllocateFlags(DebugOptions* defaults) {
  if (defaults == nullptr) {
    defaults = new DebugOptions(DefaultDebugOptionsIgnoringFlags());
  }
  flag_values = defaults;
  flag_objects = new std::vector<tsl::Flag>();
  MakeDebugOptionsFlags(flag_objects, flag_values);
  ParseFlagsFromEnvAndDieIfUnknown("ZKX_FLAGS", *flag_objects);
}

void AppendDebugOptionsFlags(std::vector<tsl::Flag>* flag_list,
                             DebugOptions* debug_options) {
  absl::call_once(flags_init, &AllocateFlags, debug_options);
  flag_list->insert(flag_list->end(), flag_objects->begin(),
                    flag_objects->end());
}

DebugOptions GetDebugOptionsFromFlags() {
  absl::call_once(flags_init, &AllocateFlags, nullptr);
  if (flag_values->zkx_flags_reset()) {
    ParseFlagsFromEnvAndDieIfUnknown("ZKX_FLAGS", *flag_objects,
                                     /*reset_envvar=*/true);
  }
  return *flag_values;
}

}  // namespace zkx
