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

#include "zkx/service/legalize_scheduling_annotations.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/hlo/ir/ptr_vec.h"
#include "zkx/side_effect_util.h"

namespace zkx {
namespace {

absl::StatusOr<int64_t> ExtractAnnotation(
    const google::protobuf::Map<std::string, std::string>& attrs,
    std::string_view instr_name) {
  int64_t annotation_id;
  if (!absl::SimpleAtoi(attrs.at(kZkxSchedulingGroupIdAttr), &annotation_id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Instruction has a non-integer scheduling annotation, inst: ",
        instr_name, ", annotation: ", attrs.at(kZkxSchedulingGroupIdAttr)));
  }
  if (annotation_id < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Instruction has a negative scheduling annotation, inst: ", instr_name,
        ", annotation: ", attrs.at(kZkxSchedulingGroupIdAttr)));
  }
  return annotation_id;
}

void DropSchedulingAnnotation(HloInstruction* instr) {
  VLOG(2) << "Dropping annotation from " << instr->name();
  FrontendAttributes frontend_attributes = instr->frontend_attributes();
  frontend_attributes.mutable_map()->erase(kZkxSchedulingGroupIdAttr);
  instr->set_frontend_attributes(frontend_attributes);
}

bool IsSupportedAsyncOp(HloInstruction* instr) {
  return HloPredicateIsOp<
      HloOpcode::kAllGatherDone, HloOpcode::kAllGatherStart,
      HloOpcode::kAllReduceDone, HloOpcode::kAllReduceStart,
      HloOpcode::kCollectivePermuteDone, HloOpcode::kCollectivePermuteStart,
      HloOpcode::kAsyncDone, HloOpcode::kAsyncStart, HloOpcode::kSendDone,
      HloOpcode::kSend, HloOpcode::kRecvDone, HloOpcode::kRecv>(instr);
}

}  // namespace

bool LegalizeSchedulingAnnotations::KeepSchedulingAnnotation(
    HloInstruction* instr) {
  return IsSupportedAsyncOp(instr) || config_.keep_sync_annotation(instr);
}

absl::StatusOr<bool> LegalizeSchedulingAnnotations::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  absl::flat_hash_map<HloInstruction*, int64_t> annotation;
  absl::flat_hash_map<
      int64_t,
      absl::flat_hash_map<HloComputation*, std::vector<HloInstruction*>>>
      annotation_to_instructions;
  // Filter the annotated ops (using config) to keep the annotations only in the
  // desired sync ops. Annotations in all async ops are kept.
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->instructions()) {
      if (!instr->frontend_attributes().map().contains(
              kZkxSchedulingGroupIdAttr) ||
          KeepSchedulingAnnotation(instr)) {
        continue;
      }
      DropSchedulingAnnotation(instr);
    }
  }
  // Find the annotated instructions and save relevant information.
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      const auto& attrs = instr->frontend_attributes().map();
      if (!attrs.contains(kZkxSchedulingGroupIdAttr)) {
        continue;
      }
      VLOG(1) << "Annotated instruction: " << instr->name() << " "
              << attrs.at(kZkxSchedulingGroupIdAttr);
      TF_ASSIGN_OR_RETURN(int64_t annotation_id,
                          ExtractAnnotation(attrs, instr->name()));

      annotation[instr] = annotation_id;
      annotation_to_instructions[annotation_id][computation].push_back(instr);
    }
  }
  // Move the annotation from inside fusion computation to the caller
  // instruction if the caller doesn't have an annotation. Return an error if
  // there are some fused instructions with different annotations.
  for (HloComputation* computation : module->computations(execution_threads)) {
    if (!computation->IsFusionComputation() ||
        !config_.keep_sync_annotation(computation->FusionInstruction()) ||
        annotation.contains(computation->FusionInstruction())) {
      continue;
    }
    int64_t seen_annotation = -1;
    for (HloInstruction* instr : computation->instructions()) {
      const auto& attrs = instr->frontend_attributes().map();
      if (!attrs.contains(kZkxSchedulingGroupIdAttr)) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(int64_t annotation_id,
                          ExtractAnnotation(attrs, instr->name()));
      if (seen_annotation == -1) {
        seen_annotation = annotation_id;
        continue;
      }
      if (seen_annotation != annotation_id) {
        return absl::InternalError(absl::StrCat(
            "Found a fusion with multiple annotations in the fused "
            "computation. fusion: ",
            computation->FusionInstruction()->name(),
            ", annotations: ", seen_annotation, " and ", annotation_id));
      }
    }
    // No fused instructions are annotated, nothing to do.
    if (seen_annotation == -1) {
      continue;
    }
    FrontendAttributes frontend_attributes =
        computation->FusionInstruction()->frontend_attributes();
    frontend_attributes.mutable_map()->insert(
        {kZkxSchedulingGroupIdAttr, std::to_string(seen_annotation)});
    computation->FusionInstruction()->set_frontend_attributes(
        frontend_attributes);
  }
  if (annotation_to_instructions.empty()) {
    return false;
  }
  absl::flat_hash_map<HloInstruction*, HloInstruction*> parent;
  for (const auto& [id, comp_inst_vector] : annotation_to_instructions) {
    for (const auto& [comp, annotated_instructions] : comp_inst_vector) {
      // First find the frontier nodes that are not annotated with id but use an
      // annotated instruction with id.
      std::vector<HloInstruction*> stack;
      absl::flat_hash_set<HloInstruction*> visited;
      for (HloInstruction* instr : annotated_instructions) {
        CHECK(annotation.contains(instr));
        CHECK_EQ(annotation[instr], id);
        if (HloPredicateIsOp<
                HloOpcode::kAllGatherDone, HloOpcode::kAllReduceDone,
                HloOpcode::kCollectivePermuteDone, HloOpcode::kAsyncDone>(
                instr) &&
            (!annotation.contains(instr->operand(0)) ||
             annotation[instr->mutable_operand(0)] != id)) {
          return absl::InternalError(absl::StrCat(
              "Done instruction's operand is not annotated with the same id: ",
              instr->operand(0)->name(), ", annotation: ", id));
        }
        for (const PtrVec<HloInstruction*>& users :
             {instr->users(), instr->control_successors()}) {
          for (HloInstruction* user : users) {
            if (!visited.contains(user) &&
                (!annotation.contains(user) || annotation[user] != id)) {
              stack.push_back(user);
              parent[user] = instr;
              visited.insert(user);
              VLOG(2) << "Annotation group: " << id
                      << ", frontier using a root: " << user->name();
            }
          }
        }
      }
      VLOG(2) << "Annotation group: " << id << ", frontier has " << stack.size()
              << " instructions";
      // Traverse the HLO graph starting from the frontier instructions and move
      // to the users. If there are gaps in the annotation, the traversal will
      // hit an instruction that is annotated with the same id.
      while (!stack.empty()) {
        HloInstruction* instr = stack.back();
        stack.pop_back();
        for (const PtrVec<HloInstruction*>& users :
             {instr->users(), instr->control_successors()}) {
          for (HloInstruction* user : users) {
            if (annotation.contains(user) && annotation[user] == id) {
              return absl::UnimplementedError(absl::StrCat(
                  "Support for annotation groups with gaps doesn't "
                  "exist yet, annotation: ",
                  id, ", instr: ", user->name(),
                  " has the same annotation in its operand tree but "
                  "has gaps on the way from that operand to itself."));
            }
            if (visited.contains(user)) {
              continue;
            }
            stack.push_back(user);
            parent[user] = instr;
            visited.insert(user);
          }
        }
      }
    }
  }
  return true;
}

}  // namespace zkx
