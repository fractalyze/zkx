/* Copyright 2019 The OpenXLA Authors.

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

#include "zkx/service/dynamic_dimension_inference.h"

#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/comparison_util.h"
#include "zkx/hlo/analysis/hlo_dataflow_analysis.h"
#include "zkx/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "zkx/hlo/ir/dynamic_parameter_binding.h"
#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
// #include "zkx/service/dynamic_window_utils.h"
// #include "zkx/service/hlo_creation_utils.h"
#include "zkx/service/hlo_value.h"
#include "zkx/service/tuple_util.h"
// #include "zkx/service/while_util.h"
#include "zkx/shape_tree.h"
#include "zkx/status_macros.h"
#include "zkx/util.h"
// #include "zkx/window_util.h"
#include "zkx/zkx_data.pb.h"

namespace zkx {

class DynamicDimensionInferenceVisitor : public DfsHloRewriteVisitor {
 public:
  explicit DynamicDimensionInferenceVisitor(
      const DynamicParameterBinding& param_bindings,
      HloDataflowAnalysis& dataflow_analysis, DynamicDimensionInference* parent,
      DynamicDimensionInference::CustomCallInferenceHandler custom_call_handler,
      DynamicDimensionInference::ShapeCheckMode shape_check_mode,
      DynamicDimensionInference::AssertionGenerator assertion_generator)
      : param_bindings_(param_bindings),
        dataflow_analysis_(dataflow_analysis),
        parent_(parent),
        custom_call_handler_(std::move(custom_call_handler)),
        shape_check_mode_(shape_check_mode),
        assertion_generator_(assertion_generator) {}

  absl::Status DefaultAction(HloInstruction* hlo) override;

  static absl::StatusOr<bool> Run(
      HloComputation* computation, HloDataflowAnalysis& dataflow_analysis,
      const DynamicParameterBinding& param_bindings,
      DynamicDimensionInference* parent,
      DynamicDimensionInference::CustomCallInferenceHandler
          custom_call_handler = nullptr,
      DynamicDimensionInference::ShapeCheckMode shape_check_mode =
          DynamicDimensionInference::ShapeCheckMode::kIgnore,
      const DynamicDimensionInference::AssertionGenerator& assertion_generator =
          nullptr) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          parent->execution_threads_)) {
      return false;
    }
    DynamicDimensionInferenceVisitor visitor(
        param_bindings, dataflow_analysis, parent,
        std::move(custom_call_handler), shape_check_mode, assertion_generator);

    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    if (visitor.shape_assertion_ != nullptr) {
      CHECK(assertion_generator);
      assertion_generator(visitor.shape_assertion_);
    }
    return visitor.changed();
  }

  absl::Status HandleParameter(HloInstruction* hlo) override;

  absl::Status HandleInfeed(HloInstruction* hlo) override;

  absl::Status HandleConstant(HloInstruction* hlo) override;

  absl::Status HandleReduce(HloInstruction* hlo) override;

  absl::Status HandleDot(HloInstruction* hlo) override;

  absl::Status HandleTuple(HloInstruction* hlo) override;

  absl::Status HandleTranspose(HloInstruction* hlo) override;

  absl::Status HandleDynamicReshape(HloInstruction* hlo) override;

  absl::Status HandleReshape(HloInstruction* hlo) override;

  // TODO(chokobole): Uncomment this. Dependency: HandleSort
  // absl::Status HandleSort(HloInstruction* hlo) override;

  // TODO(chokobole): Uncomment this. Dependency: HandlePad
  // absl::Status HandlePad(HloInstruction* hlo) override;

  absl::Status HandleCustomCall(HloInstruction* hlo) override;

  absl::Status HandleBroadcast(HloInstruction* hlo) override;

  absl::Status HandleGetDimensionSize(HloInstruction* hlo) override;

  absl::Status HandleSetDimensionSize(HloInstruction* hlo) override;

  absl::Status HandleSelect(HloInstruction* hlo) override;

  absl::Status HandleConcatenate(HloInstruction* hlo) override;

  absl::Status HandleReverse(HloInstruction* hlo) override;

  absl::Status HandleGetTupleElement(HloInstruction* hlo) override;

  absl::Status HandleElementwiseUnary(HloInstruction* hlo) override;

  absl::Status HandleElementwiseNary(HloInstruction* hlo);

  absl::Status HandleElementwiseBinary(HloInstruction* hlo) override;

  // TODO(chokobole): Uncomment this. Dependency: HandleClamp
  // absl::Status HandleClamp(HloInstruction* hlo) override;

  absl::Status HandleConditional(HloInstruction* hlo) override;

  absl::Status HandleWhile(HloInstruction* hlo) override;

  absl::Status HandleSlice(HloInstruction* hlo) override;

  absl::Status HandleDynamicSlice(HloInstruction* hlo) override;

  absl::Status HandleDynamicUpdateSlice(HloInstruction* hlo) override;

  absl::Status HandleGather(HloInstruction* hlo) override;

  absl::Status HandleScatter(HloInstruction* hlo) override;

  absl::Status HandleMap(HloInstruction* hlo) override;

  absl::Status HandleDomain(HloInstruction* hlo) override;

  absl::Status HandleAsyncStart(HloInstruction* hlo) override;

  absl::Status HandleAsyncDone(HloInstruction* hlo) override;

 private:
  using OperandDynamicDimensionFn = absl::FunctionRef<absl::Status(
      HloInstruction* operand, ShapeIndex index, int64_t dimension,
      int64_t operand_index, HloInstruction* dynamic_size)>;

  using DynamicDimensionFn = std::function<absl::Status(
      ShapeIndex index, int64_t dimension, HloInstruction* dynamic_size)>;

  void SetDynamicSize(HloInstruction* inst, const ShapeIndex& index,
                      int64_t dim, HloInstruction* size,
                      bool clear_dynamic_dimension = true);

  void SetDynamicSizes(HloInstruction* inst, const ShapeIndex& index,
                       absl::Span<HloInstruction* const> sizes);

  absl::Status HandleDynamicWindowSamePadding(HloInstruction* hlo,
                                              HloInstruction* dynamic_size,
                                              int64_t operand_index,
                                              int64_t dimension);

  absl::Status ForEachOperandDynamicDimension(HloInstruction* inst,
                                              OperandDynamicDimensionFn);
  absl::Status ForEachDynamicDimensionInOperand(HloInstruction* inst,
                                                int64_t operand_index,
                                                OperandDynamicDimensionFn);
  absl::Status ForEachDynamicDimension(HloInstruction* inst,
                                       const DynamicDimensionFn& fn);

  bool CanInfer(HloInstruction* hlo) { return parent_->CanInfer(hlo); }

  // Return true unless all users of the instruction can consume a dynamic shape
  // (including uses across control flow, but only within the same thread). The
  // given `ShapeIndex` is the leaf array returned by the given instruction that
  // will be considered.
  absl::StatusOr<bool> RequiresPadToStatic(HloInstruction* instr,
                                           ShapeIndex shape_index);

  // Insert pad-to-static after `inst` if `inst` has dynamic dimensions in it
  // and `RequiresPadToStatic` is true for all leaves. If the instruction
  // produces a tuple, each tuple component will be considered independently.
  // Returns the original instruction, with all arrays converted to static
  // shapes.
  absl::Status InsertPadToStaticOnInstruction(HloInstruction* inst);

  // Insert shape check to make sure `dim1` is equal to `dim2`. If
  // support_implicit_broadcast is true, the check will pass if either of them
  // is 1, even if they are different.
  absl::Status InsertShapeCheck(HloInstruction* dim1, HloInstruction* dim2,
                                bool support_implicit_broadcast);

  // Pass through a dynamic dimension from the input to the output with the
  // same value and index in the shape. This is a helper function to handle
  // trivial instructions like elementwise operations.
  absl::Status PassThroughDynamicDimension(HloInstruction*);

  // The dynamic parameter bindings of this computation.
  const DynamicParameterBinding& param_bindings_;

  HloDataflowAnalysis& dataflow_analysis_;

  // A pointer to DynamicDimensionInference, used to update the dynamic mapping.
  DynamicDimensionInference* parent_;  // not owned

  // A handler for custom calls.
  DynamicDimensionInference::CustomCallInferenceHandler custom_call_handler_;

  // Indicates what to do at places where shape check is needed.
  DynamicDimensionInference::ShapeCheckMode shape_check_mode_;

  // Value which has to be `true` for the shapes to match.
  HloInstruction* shape_assertion_ = nullptr;  // not owned

  DynamicDimensionInference::AssertionGenerator assertion_generator_;
};

void DynamicDimensionInferenceVisitor::SetDynamicSize(
    HloInstruction* inst, const ShapeIndex& index, int64_t dim,
    HloInstruction* size, bool clear_dynamic_dimension) {
  parent_->SetDynamicSize(inst, index, dim, size);
  // Clear the dynamic dimension since we have recorded a dynamic size.
  // If there are any dynamic dimensions left after DynamicPadder has completely
  // run, we will raise an error.
  if (clear_dynamic_dimension) {
    ShapeUtil::GetMutableSubshape(inst->mutable_shape(), index)
        ->set_dynamic_dimension(dim, false);
  }
  MarkAsChanged();
}

void DynamicDimensionInferenceVisitor::SetDynamicSizes(
    HloInstruction* inst, const ShapeIndex& index,
    absl::Span<HloInstruction* const> sizes) {
  const Shape& subshape = ShapeUtil::GetSubshape(inst->shape(), index);
  CHECK(subshape.IsArray() && subshape.rank() == sizes.size());
  for (int64_t dimension = 0; dimension < subshape.rank(); ++dimension) {
    if (sizes[dimension] != nullptr) {
      SetDynamicSize(inst, index, dimension, sizes[dimension]);
    }
  }
}

absl::Status DynamicDimensionInferenceVisitor::DefaultAction(
    HloInstruction* hlo) {
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
               int64_t operand_index, HloInstruction* dynamic_size) {
        return absl::UnimplementedError(absl::StrCat(
            "Asked to propagate a dynamic dimension from hlo ", operand->name(),
            "@", index.ToString(), "@", dimension, " to hlo ", hlo->ToString(),
            ", which is not implemented."));
      });
}

absl::Status DynamicDimensionInferenceVisitor::HandleGetTupleElement(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> absl::Status {
        if (hlo->tuple_index() != index[0]) {
          return absl::OkStatus();
        }
        ShapeIndex new_index(ShapeIndexView(index).subspan(1));
        SetDynamicSize(hlo, new_index, dimension, dynamic_size);
        return absl::OkStatus();
      });
}

absl::Status DynamicDimensionInferenceVisitor::HandleTuple(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction*, ShapeIndex index, int64_t dimension,
               int64_t operand_index, HloInstruction* dynamic_size) {
        index.push_front(operand_index);
        SetDynamicSize(hlo, index, dimension, dynamic_size);
        return absl::OkStatus();
      }));
  return absl::OkStatus();
}

absl::Status DynamicDimensionInferenceVisitor::HandleBroadcast(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
               int64_t operand_index, HloInstruction* dynamic_size) {
        int64_t broadcast_dim = hlo->dimensions(dimension);
        SetDynamicSize(hlo, {}, broadcast_dim, dynamic_size);
        return absl::OkStatus();
      });
}

absl::Status DynamicDimensionInferenceVisitor::HandleConstant(
    HloInstruction* hlo) {
  if (!hlo->shape().is_dynamic()) {
    return absl::OkStatus();
  }
  auto* constant = Cast<HloConstantInstruction>(hlo);
  ShapeTree<bool> do_pad(constant->shape(), false);
  Shape padded_shape = constant->shape();
  bool pad_any = false;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachMutableSubshapeWithStatus(
      &padded_shape,
      [&](Shape* subshape, const ShapeIndex& index) -> absl::Status {
        if (!subshape->IsArray()) {
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(bool requires_pad, RequiresPadToStatic(hlo, index));
        if (requires_pad) {
          pad_any = *do_pad.mutable_element(index) = true;
          *subshape = ShapeUtil::MakeStaticShape(*subshape);
        }
        return absl::OkStatus();
      }));
  if (!pad_any) {
    return absl::OkStatus();
  }
  Literal padded_literal(padded_shape);
  do_pad.ForEachElement([&](const ShapeIndex& index, bool requires_pad) {
    const Shape& subshape = ShapeUtil::GetSubshape(padded_shape, index);
    if (!subshape.IsArray()) {
      return absl::OkStatus();
    }
    TF_RETURN_IF_ERROR(padded_literal.CopyFrom(constant->literal(), index,
                                               index,
                                               /*only_dynamic_bound=*/true));
    if (!requires_pad) {
      for (int64_t dimension = 0; dimension < subshape.rank(); ++dimension) {
        if (subshape.is_dynamic_dimension(dimension)) {
          padded_literal.SetDynamicSize(
              dimension, index,
              constant->literal().GetDynamicSize(dimension, index));
        }
      }
    }
    return absl::OkStatus();
  });
  auto* padded_constant = hlo->AddInstruction(
      HloInstruction::CreateConstant(std::move(padded_literal)));
  TF_RETURN_IF_ERROR(constant->ReplaceAllUsesWith(padded_constant));
  SetVisited(*padded_constant);
  TF_RETURN_IF_ERROR(do_pad.ForEachElementWithStatus(
      [&](const ShapeIndex& index, bool requires_pad) -> absl::Status {
        if (!requires_pad) {
          return absl::OkStatus();
        }
        const Shape& subshape =
            ShapeUtil::GetSubshape(constant->shape(), index);
        TF_RET_CHECK(subshape.IsArray());
        for (int64_t dimension = 0; dimension < subshape.rank(); ++dimension) {
          if (!subshape.is_dynamic_dimension(dimension)) {
            continue;
          }
          HloInstruction* dynamic_size = hlo->AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
                  constant->literal().GetDynamicSize(dimension, index))));
          SetVisited(*dynamic_size);
          SetDynamicSize(padded_constant, index, dimension, dynamic_size);
        }
        return absl::OkStatus();
      }));
  MarkAsChanged();
  return absl::OkStatus();
}

absl::Status DynamicDimensionInferenceVisitor::HandleCustomCall(
    HloInstruction* hlo) {
  // clang-format off
  // TODO(chokobole): Implement this. Dependency: HloInstruction::custom_call_target
  // clang-format on
  return absl::UnimplementedError(
      "DynamicDimensionInferenceVisitor::HandleCustomCall not implemented");
}

absl::Status DynamicDimensionInferenceVisitor::HandleReduce(
    HloInstruction* hlo) {
  // TODO(chokobole): Implement this. Dependency: HloReduceInstruction
  return absl::UnimplementedError(
      "DynamicDimensionInferenceVisitor::HandleReduce not implemented");
}

absl::Status DynamicDimensionInferenceVisitor::HandleDot(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  absl::InlinedVector<HloInstruction*, 4> dynamic_sizes(hlo->shape().rank(),
                                                        nullptr);
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex operand_shape_index,
          int64_t operand_dimension, int64_t operand_index,
          HloInstruction* dynamic_size) -> absl::Status {
        // There are three types of dimensions in a dot:
        // A. batch dims
        // B. contracting dims
        // C. non-batch non-contracting dims.
        // The output dimensions of a dot has three parts with the following
        // order:
        // [(type A), (lhs type C), (rhs type C)]
        //
        // Note that both lhs and rhs have the same dimension sizes for batch,
        // but the dimension index could be different.
        //
        // Given one dynamic input dimension, either lhs or rhs, we use a
        // mapping to find the corresponding output dimension.
        HloInstruction* dot = hlo;
        const DotDimensionNumbers& dimension_numbers =
            dot->dot_dimension_numbers();
        // A map from the operand dimensions to result dimension.
        absl::flat_hash_map<int64_t, int64_t> result_dim_mapping;
        int64_t current_result_dims = 0;

        bool lhs = operand_index == 0;

        // The first loop keep tracks of batch dimension. RHS and LHS could have
        // different batch dimension numbers.
        if (lhs) {
          for (int64_t i : dimension_numbers.lhs_batch_dimensions()) {
            result_dim_mapping[i] = current_result_dims++;
          }
        } else {
          for (int64_t i : dimension_numbers.rhs_batch_dimensions()) {
            result_dim_mapping[i] = current_result_dims++;
          }
        }

        // Handle dimensions in the lhs.
        for (int64_t i = 0; i < dot->operand(0)->shape().rank(); i++) {
          // Look for non-contracting and non-batching dimension.
          if (absl::c_linear_search(
                  dimension_numbers.lhs_contracting_dimensions(), i)) {
            continue;
          }
          if (absl::c_linear_search(dimension_numbers.lhs_batch_dimensions(),
                                    i)) {
            continue;
          }
          if (lhs) {
            result_dim_mapping[i] = current_result_dims;
          }
          current_result_dims++;
        }

        // Handle dimensions in the rhs.
        for (int64_t i = 0; i < dot->operand(1)->shape().rank(); i++) {
          // Look for non-contracting and non-batching dimension.
          if (absl::c_linear_search(
                  dimension_numbers.rhs_contracting_dimensions(), i)) {
            continue;
          }
          if (absl::c_linear_search(dimension_numbers.rhs_batch_dimensions(),
                                    i)) {
            continue;
          }
          if (!lhs) {
            result_dim_mapping[i] = current_result_dims;
          }
          current_result_dims++;
        }

        // Check if the operand dim is in the result shape. If so, add another
        // work item to trace that dimension.
        auto iter = result_dim_mapping.find(operand_dimension);
        if (iter != result_dim_mapping.end()) {
          dynamic_sizes[iter->second] = dynamic_size;
        }

        return absl::OkStatus();
      }));

  SetDynamicSizes(hlo, {}, dynamic_sizes);

  return absl::OkStatus();
}

absl::Status DynamicDimensionInferenceVisitor::HandleTranspose(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> absl::Status {
        int64_t permuted_dim = -1;
        for (int64_t i = 0; i < hlo->dimensions().size(); ++i) {
          if (hlo->dimensions()[i] == dimension) {
            TF_RET_CHECK(permuted_dim == -1);
            permuted_dim = i;
          }
        }
        SetDynamicSize(hlo, {}, permuted_dim, dynamic_size);
        return absl::OkStatus();
      });
}

absl::Status DynamicDimensionInferenceVisitor::HandleConcatenate(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  // First handle concatenate dimensions. We do this by iterating through all
  // operands while tracking both dynamic and static dimensions.

  // static_size is used to keep track of the concatenated size of static
  // dimensions.
  int64_t static_size = 0;
  std::vector<HloInstruction*> dynamic_concat_dims;
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    HloInstruction* concat_dim_size = nullptr;
    for (int64_t dimension = 0; dimension < hlo->operand(i)->shape().rank();
         ++dimension) {
      if (dimension == hlo->concatenate_dimension()) {
        HloInstruction* dynamic_size =
            parent_->GetDynamicSize(hlo->mutable_operand(i), {}, dimension);
        concat_dim_size = dynamic_size;
      }
    }
    if (concat_dim_size == nullptr) {
      // This is a static dimension.
      static_size +=
          hlo->operand(i)->shape().dimensions(hlo->concatenate_dimension());
    } else {
      dynamic_concat_dims.push_back(concat_dim_size);
    }
  }
  // If concat dimension is dynamic, calculate its size by summing up static
  // dims and dynamic dims together.
  std::vector<HloInstruction*> dynamic_sizes(hlo->shape().rank(), nullptr);
  if (!dynamic_concat_dims.empty()) {
    HloInstruction* dim_size_total =
        hlo->parent()->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(static_size)));
    for (HloInstruction* dynamic_dim : dynamic_concat_dims) {
      dim_size_total = hlo->parent()->AddInstruction(
          HloInstruction::CreateBinary(dim_size_total->shape(), HloOpcode::kAdd,
                                       dim_size_total, dynamic_dim));
    }
    dynamic_sizes[hlo->concatenate_dimension()] = dim_size_total;
  }

  // Simply pass through non-concat dynamic dimensions.
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> absl::Status {
        TF_RET_CHECK(index.empty());
        int64_t concatenate_dimension = hlo->concatenate_dimension();
        if (concatenate_dimension == dimension) {
          return absl::OkStatus();
        }
        dynamic_sizes[dimension] = dynamic_size;
        return absl::OkStatus();
      }));

  SetDynamicSizes(hlo, {}, dynamic_sizes);

  return absl::OkStatus();
}

absl::Status DynamicDimensionInferenceVisitor::HandleGetDimensionSize(
    HloInstruction* gds) {
  // TODO(chokobole): Implement this. Dependency: HloInstruction::dimension
  return absl::UnimplementedError(
      "DynamicDimensionInferenceVisitor::HandleGetDimensionSize not "
      "implemented");
}

absl::Status DynamicDimensionInferenceVisitor::HandleSetDimensionSize(
    HloInstruction* hlo) {
  // TODO(chokobole): Implement this. Dependency: HloInstruction::dimension
  return absl::UnimplementedError(
      "DynamicDimensionInferenceVisitor::HandleSetDimensionSize not "
      "implemented");
}

absl::Status DynamicDimensionInferenceVisitor::PassThroughDynamicDimension(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  // TODO(b/298671312): This is ambiguous with respect to which operand provides
  // the dynamic size.
  ShapeTree<absl::InlinedVector<HloInstruction*, 2>> dynamic_sizes(
      hlo->shape());
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo, [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
               int64_t operand_index, HloInstruction* dynamic_size) {
        const Shape& subshape = ShapeUtil::GetSubshape(hlo->shape(), index);
        auto* element = dynamic_sizes.mutable_element(index);
        element->resize(subshape.rank(), nullptr);
        (*element)[dimension] = dynamic_size;
        return absl::OkStatus();
      }));
  dynamic_sizes.ForEachElement([&](const ShapeIndex& index, const auto& sizes) {
    if (sizes.empty()) {
      return;
    }
    SetDynamicSizes(hlo, index, sizes);
  });
  return absl::OkStatus();
}

absl::Status DynamicDimensionInferenceVisitor::HandleDomain(
    HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

absl::Status DynamicDimensionInferenceVisitor::HandleAsyncStart(
    HloInstruction* hlo) {
  if (!HloInstruction::IsThreadIncluded(hlo->async_execution_thread(),
                                        parent_->execution_threads_)) {
    // Async-start not included in specified execution thread set will use
    // metadata-prefix version of dynamic shapes (result of slice-to-dynamic) so
    // there is no need to propagate dynamic dimension info.
    return absl::OkStatus();
  }
  return DefaultAction(hlo);
}

absl::Status DynamicDimensionInferenceVisitor::HandleAsyncDone(
    HloInstruction* hlo) {
  if (!HloInstruction::IsThreadIncluded(hlo->async_execution_thread(),
                                        parent_->execution_threads_)) {
    // Other threads can return a dynamic shape directly, so we may need to
    // insert PadToStatic.
    return InsertPadToStaticOnInstruction(hlo);
  }
  return DefaultAction(hlo);
}

absl::Status DynamicDimensionInferenceVisitor::HandleElementwiseUnary(
    HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

absl::Status DynamicDimensionInferenceVisitor::HandleSelect(
    HloInstruction* hlo) {
  return HandleElementwiseNary(hlo);
}

absl::Status DynamicDimensionInferenceVisitor::HandleElementwiseNary(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  // First find all the dynamic sizes of the operands, and arrange them by
  // dimension.
  absl::InlinedVector<absl::InlinedVector<HloInstruction*, 2>, 2> operand_sizes(
      hlo->shape().rank(),
      absl::InlinedVector<HloInstruction*, 2>(hlo->operand_count(), nullptr));
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> absl::Status {
        TF_RET_CHECK(index.empty());
        operand_sizes[dimension][operand_index] = dynamic_size;
        return absl::OkStatus();
      }));

  absl::InlinedVector<HloInstruction*, 2> existing_sizes(hlo->shape().rank(),
                                                         nullptr);
  for (int operand_index = 0; operand_index < hlo->operand_count();
       ++operand_index) {
    for (int64_t dimension = 0; dimension < hlo->shape().rank(); ++dimension) {
      HloInstruction* dynamic_size = operand_sizes[dimension][operand_index];
      if (dynamic_size == nullptr) {
        continue;
      }
      HloInstruction* existing_size = existing_sizes[dimension];
      if (existing_size == nullptr) {
        existing_sizes[dimension] = dynamic_size;
      } else if (existing_sizes[dimension] != dynamic_size) {
        return absl::UnimplementedError(
            "DynamicDimensionInferenceVisitor::HandleElementwiseNary not "
            "implemented when operand dynamic size is different from existing "
            "size");
        // TF_RETURN_IF_ERROR(
        //     InsertShapeCheck(existing_size, dynamic_size,
        //                      /*support_implicit_broadcast=*/true));

        // auto one = comp->AddInstruction(
        //     HloInstruction::CreateConstant(LiteralUtil::One(S32)));

        // auto operand_needs_broadcast =
        //     comp->AddInstruction(HloInstruction::CreateCompare(
        //         ShapeUtil::MakeShape(PRED, {}), dynamic_size, existing_size,
        //         ComparisonDirection::kLt));
        // auto is_one = comp->AddInstruction(HloInstruction::CreateCompare(
        //     ShapeUtil::MakeShape(PRED, {}), dynamic_size, one,
        //     ComparisonDirection::kEq));
        // operand_needs_broadcast =
        //     comp->AddInstruction(HloInstruction::CreateBinary(
        //         ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, is_one,
        //         operand_needs_broadcast));

        // auto existing_needs_broadcast =
        //     comp->AddInstruction(HloInstruction::CreateCompare(
        //         ShapeUtil::MakeShape(PRED, {}), existing_size, dynamic_size,
        //         ComparisonDirection::kLt));
        // is_one = comp->AddInstruction(HloInstruction::CreateCompare(
        //     ShapeUtil::MakeShape(PRED, {}), existing_size, one,
        //     ComparisonDirection::kEq));
        // existing_needs_broadcast =
        //     comp->AddInstruction(HloInstruction::CreateBinary(
        //         ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, is_one,
        //         existing_needs_broadcast));

        // auto needs_broadcast =
        //     comp->AddInstruction(HloInstruction::CreateBinary(
        //         ShapeUtil::MakeShape(PRED, {}), HloOpcode::kOr,
        //         operand_needs_broadcast, existing_needs_broadcast));
        // auto max_size = comp->AddInstruction(HloInstruction::CreateBinary(
        //     ShapeUtil::MakeScalarShape(S32), HloOpcode::kMaximum,
        //     dynamic_size, existing_size));
        // auto min_size = comp->AddInstruction(HloInstruction::CreateBinary(
        //     ShapeUtil::MakeScalarShape(S32), HloOpcode::kMinimum,
        //     dynamic_size, existing_size));
        // auto select_size =
        // comp->AddInstruction(HloInstruction::CreateTernary(
        //     ShapeUtil::MakeScalarShape(S32), HloOpcode::kSelect,
        //     needs_broadcast, max_size, min_size));
        // existing_sizes[dimension] = select_size;
      }
    }
  }

  SetDynamicSizes(hlo, {}, existing_sizes);

  return absl::OkStatus();
}

absl::Status DynamicDimensionInferenceVisitor::HandleElementwiseBinary(
    HloInstruction* hlo) {
  return HandleElementwiseNary(hlo);
}

absl::Status DynamicDimensionInferenceVisitor::HandleDynamicReshape(
    HloInstruction* hlo) {
  // TODO(chokobole): Implement this. Dependency: HloDynamicReshapeInstruction
  return absl::UnimplementedError(
      "DynamicDimensionInferenceVisitor::HandleDynamicReshape not implemented");
}

absl::Status DynamicDimensionInferenceVisitor::HandleReshape(
    HloInstruction* const hlo) {
  // clang-format off
  // TODO(chokobole): Implement this. Dependency: HloInstruction::inferred_dimension
  // clang-format on
  return absl::UnimplementedError(
      "DynamicDimensionInferenceVisitor::HandleReshape not implemented");
}

absl::Status DynamicDimensionInferenceVisitor::HandleSlice(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex /*index*/, int64_t dimension,
          int64_t /*operand_index*/,
          HloInstruction* dynamic_size) -> absl::Status {
        int64_t start = hlo->slice_starts(dimension);
        int64_t limit = hlo->slice_limits(dimension);
        int64_t stride = hlo->slice_strides(dimension);
        int64_t size = CeilOfRatio<int64_t>(limit - start, stride);
        if (size == 1) {
          TF_RET_CHECK(!hlo->shape().is_dynamic_dimension(dimension));
          // Slicing a single element out eliminates the dynamic dimension.
          return absl::OkStatus();
        }

        TF_RET_CHECK(hlo->shape().is_dynamic_dimension(dimension));
        if (start != 0) {
          dynamic_size = hlo->AddInstruction(HloInstruction::CreateBinary(
              dynamic_size->shape(), HloOpcode::kSubtract, dynamic_size,
              hlo->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(start)))));
        }
        if (stride != 1) {
          dynamic_size = hlo->AddInstruction(HloInstruction::CreateBinary(
              dynamic_size->shape(), HloOpcode::kAdd, dynamic_size,
              hlo->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(stride - 1)))));
          dynamic_size = hlo->AddInstruction(HloInstruction::CreateBinary(
              dynamic_size->shape(), HloOpcode::kDivide, dynamic_size,
              hlo->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32_t>(stride)))));
        }
        SetDynamicSize(hlo, {}, dimension, dynamic_size);

        return absl::OkStatus();
      });
}

absl::Status DynamicDimensionInferenceVisitor::HandleDynamicSlice(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  return ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> absl::Status {
        // Slicing a single element out kills the dynamic dimension.
        if (hlo->shape().dimensions(dimension) == 1) {
          return absl::OkStatus();
        }
        if (hlo->shape().dimensions(dimension) !=
            hlo->operand(0)->shape().dimensions(dimension)) {
          return absl::UnimplementedError(absl::StrFormat(
              "Dynamic dimension propagation on DynamicSlice where a partial "
              "dimension is selected %s",
              hlo->ToString()));
        }

        // Only the base operand should be dynamic (since the rest are scalars).
        TF_RET_CHECK(operand_index == 0);

        TF_RET_CHECK(index.empty());
        SetDynamicSize(hlo, {}, dimension, dynamic_size);

        return absl::OkStatus();
      });
}

absl::Status DynamicDimensionInferenceVisitor::HandleDynamicUpdateSlice(
    HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  absl::InlinedVector<HloInstruction*, 2> output_dynamic_sizes(
      hlo->shape().rank(), nullptr);
  TF_RETURN_IF_ERROR(ForEachOperandDynamicDimension(
      hlo,
      [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
          int64_t operand_index, HloInstruction* dynamic_size) -> absl::Status {
        TF_RET_CHECK(index.empty());

        if (hlo->shape().dimensions(dimension) !=
            hlo->operand(0)->shape().dimensions(dimension)) {
          return absl::UnimplementedError(absl::StrFormat(
              "Dynamic dimension propagation on DynamicUpdateSlice where a "
              "partial dimension is selected %s",
              hlo->ToString()));
        }

        if (operand_index == 1 &&
            hlo->operand(1)->shape().dimensions(dimension) <
                hlo->operand(0)->shape().dimensions(dimension)) {
          // DUS(input=[A], update=[<=B])
          //
          // If update dim is smaller than input dim (B < A), then we are doing
          // a partial update, no need to set the output dynamic dimension.
          //
          // The dynamic shape in `update` doesn't change output dynamic shape.
          hlo->mutable_shape()->set_dynamic_dimension(dimension, false);
          return absl::OkStatus();
        }

        output_dynamic_sizes[dimension] = dynamic_size;

        return absl::OkStatus();
      }));
  SetDynamicSizes(hlo, {}, output_dynamic_sizes);
  return absl::OkStatus();
}

absl::Status DynamicDimensionInferenceVisitor::HandleReverse(
    HloInstruction* hlo) {
  return PassThroughDynamicDimension(hlo);
}

absl::Status DynamicDimensionInferenceVisitor::HandleGather(
    HloInstruction* hlo) {
  // TODO(chokobole): Implement this. Dependency: HloGatherInstruction
  return absl::UnimplementedError(
      "DynamicDimensionInferenceVisitor::HandleGather not implemented");
}

absl::Status DynamicDimensionInferenceVisitor::HandleConditional(
    HloInstruction* hlo) {
  // TODO(chokobole): Implement this. Dependency: CallInliner
  return absl::UnimplementedError(
      "DynamicDimensionInferenceVisitor::HandleConditional not implemented");
}

absl::Status DynamicDimensionInferenceVisitor::HandleMap(HloInstruction* hlo) {
  if (!CanInfer(hlo)) {
    return absl::OkStatus();
  }
  return HandleElementwiseNary(hlo);
}

absl::Status DynamicDimensionInferenceVisitor::HandleScatter(
    HloInstruction* hlo) {
  // TODO(chokobole): Implement this. Dependency: HloScatterInstruction
  return absl::UnimplementedError(
      "DynamicDimensionInferenceVisitor::HandleScatter not implemented");
}

absl::Status DynamicDimensionInferenceVisitor::HandleWhile(
    HloInstruction* hlo) {
  // TODO(chokobole): Implement this. Dependency: WhileUtil
  return absl::UnimplementedError(
      "DynamicDimensionInferenceVisitor::HandleWhile not implemented");
}

absl::Status DynamicDimensionInferenceVisitor::HandleParameter(
    HloInstruction* hlo) {
  if (hlo->parent()->IsEntryComputation()) {
    TF_RET_CHECK(param_bindings_.empty());
    return InsertPadToStaticOnInstruction(hlo);
  }

  return param_bindings_.ForEachBinding(
      [&](const DynamicParameterBinding::DynamicSizeParameter& dynamic_size,
          const DynamicParameterBinding::DynamicDimension& dynamic_dimension)
          -> absl::Status {
        if (dynamic_dimension.parameter_num == hlo->parameter_number()) {
          SetDynamicSize(
              hlo, dynamic_dimension.parameter_index,
              dynamic_dimension.dimension,
              TupleUtil::AddGetTupleElements(HloPosition{
                  /*instruction=*/hlo->parent()->parameter_instruction(
                      dynamic_size.parameter_num),
                  /*index=*/dynamic_size.parameter_index,
              }));
        }
        return absl::OkStatus();
      });
}

absl::Status DynamicDimensionInferenceVisitor::HandleInfeed(
    HloInstruction* hlo) {
  return InsertPadToStaticOnInstruction(hlo);
}

absl::Status DynamicDimensionInferenceVisitor::ForEachDynamicDimension(
    HloInstruction* inst, const DynamicDimensionFn& fn) {
  auto iter = parent_->per_hlo_dynamic_dimensions_.find(inst);
  if (iter != parent_->per_hlo_dynamic_dimensions_.end()) {
    for (auto& dynamic_dimension : iter->second) {
      HloInstruction* dynamic_size = parent_->GetDynamicSize(
          dynamic_dimension.inst, dynamic_dimension.index,
          dynamic_dimension.dim);
      TF_RETURN_IF_ERROR(
          fn(dynamic_dimension.index, dynamic_dimension.dim, dynamic_size));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> DynamicDimensionInferenceVisitor::RequiresPadToStatic(
    HloInstruction* instr, ShapeIndex shape_index) {
  TF_RET_CHECK(ShapeUtil::IsLeafIndex(instr->shape(), shape_index))
      << instr->shape() << " @ " << shape_index;
  if (ShapeUtil::GetSubshape(instr->shape(), shape_index).is_static()) {
    return false;
  }
  auto uses =
      dataflow_analysis_.GetValueDefinedAt(instr, shape_index).GetUses();
  for (const auto& use : uses) {
    if (use.instruction->opcode() == HloOpcode::kAsyncStart ||
        use.instruction->opcode() == HloOpcode::kAsyncUpdate ||
        use.instruction->opcode() == HloOpcode::kAsyncDone ||
        use.instruction->opcode() == HloOpcode::kCall ||
        use.instruction->opcode() == HloOpcode::kTuple ||
        use.instruction->opcode() == HloOpcode::kGetTupleElement ||
        use.instruction->opcode() == HloOpcode::kConditional) {
      // These uses do not require padding as they do not operate the data.
      continue;
    }
    if (use.instruction->opcode() == HloOpcode::kWhile) {
      TF_RET_CHECK(use.operand_number == 0);
      HloInstruction* root = use.instruction->while_body()->root_instruction();
      if (parent_->HasDynamicDimension(root, use.operand_index)) {
        return true;
      }
      continue;
    }
    if (use.instruction->opcode() == HloOpcode::kSetDimensionSize) {
      // The dynamic size cannot itself be dynamic.
      TF_RET_CHECK(use.operand_number == 0);
      // SetDimensionSize will be removed, so the array must be padded if it
      // is a user of the array.
      return true;
    }
    if (use.instruction->opcode() == HloOpcode::kGetDimensionSize) {
      return true;
    }
    // TODO(chokobole): Uncomment this. Dependency: HloInstruction::IsCustomCall
    // if (use.instruction->opcode() != HloOpcode::kCustomCall ||
    //     !use.instruction->IsCustomCall({"PadToStatic", "Sharding",
    //                                     "SPMDShardToFullShape",
    //                                     "SPMDFullToShardShape"})) {
    //   if (parent_->op_supports_dynamism_handler_ == nullptr) {
    //     return true;
    //   }
    //   if (parent_->op_supports_dynamism_handler_(use.instruction) ==
    //       OpDynamismSupport::kNoSupport) {
    //     return true;
    //   }
    // }
  }

  // Don't do pad-to-static.
  return false;
}

// Insert pad-to-static after `inst` if `inst` has dynamic dimensions in it.
// If the instruction produces a tuple, each tuple component will be considered
// independently.
absl::Status DynamicDimensionInferenceVisitor::InsertPadToStaticOnInstruction(
    HloInstruction* inst) {
  if (inst->shape().is_static()) {
    return absl::OkStatus();
  }

  // Decide while leaf arrays need to be padded.
  ShapeTree<bool> needs_pad(inst->shape(), false);
  bool any_needs_pad = false;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      inst->shape(), [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (subshape.IsTuple()) {
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(bool do_pad,
                            RequiresPadToStatic(inst, shape_index));
        if (do_pad) {
          *needs_pad.mutable_element(shape_index) = true;
          any_needs_pad = true;
        }
        return absl::OkStatus();
      }));

  if (!any_needs_pad) {
    return absl::OkStatus();
  }

  auto users = inst->users();

  ShapeTree<HloInstruction*> gtes =
      TupleUtil::DisassembleTupleInstruction(inst);

  // Add PadToStatic to the leaf arrays and record the dynamic dimensions.
  ShapeTree<HloInstruction*> padded(inst->shape(), nullptr);
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapePostOrderWithStatus(
      inst->shape(),
      [&](const Shape& subshape,
          const ShapeIndex& shape_index) -> absl::Status {
        HloInstruction* element = gtes.element(shape_index);
        SetVisited(*gtes.element(shape_index));
        if (subshape.IsTuple()) {
          absl::InlinedVector<HloInstruction*, 2> children;
          ShapeIndex child_index = shape_index;
          for (int i = 0; i < subshape.tuple_shapes_size(); ++i) {
            child_index.push_back(i);
            children.push_back(padded.element(child_index));
            child_index.pop_back();
          }
          HloInstruction* tuple =
              element->AddInstruction(HloInstruction::CreateVariadic(
                  subshape, HloOpcode::kTuple, children));
          CHECK_OK(ForEachOperandDynamicDimension(
              tuple,
              [&](HloInstruction* operand, ShapeIndex index, int64_t dimension,
                  int64_t operand_index, HloInstruction* dynamic_size) {
                index.push_front(operand_index);
                SetDynamicSize(tuple, index, dimension, dynamic_size);
                return absl::OkStatus();
              }));
          *padded.mutable_element(shape_index) = tuple;
          return absl::OkStatus();
        }
        if (needs_pad.element(shape_index)) {
          // clang-format off
          // TODO(chokobole): Implement this. Dependency: HloInstruction::CreateCustomCall
          // clang-format on
          return absl::UnimplementedError(
              "DynamicDimensionInferenceVisitor::"
              "InsertPadToStaticOnInstruction "
              "not implemented when needs_pad is true");
        } else {
          *padded.mutable_element(shape_index) = element;
        }
        return absl::OkStatus();
      }));

  HloInstruction* result = padded.element({});

  // Replace all uses of the original instruction with the padded outputs.
  for (auto user : users) {
    for (int64_t i : user->OperandIndices(inst)) {
      TF_RETURN_IF_ERROR(user->ReplaceOperandWith(i, result));
    }
  }
  if (inst->IsRoot()) {
    inst->parent()->set_root_instruction(result);
  }

  MarkAsChanged();

  return absl::OkStatus();
}

absl::Status DynamicDimensionInferenceVisitor::InsertShapeCheck(
    HloInstruction* dim1, HloInstruction* dim2,
    bool support_implicit_broadcast) {
  switch (shape_check_mode_) {
    case DynamicDimensionInference::kIgnore:
      return absl::OkStatus();
    case DynamicDimensionInference::kCompileTime:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Fail to proof the equality of two dimensions at compile time: "
          "%s vs %s",
          dim1->ToString(), dim2->ToString()));
    case DynamicDimensionInference::kRuntime: {
      // clang-format off
      // TODO(chokobole): Uncomment this. Dependency: MakeCompareHlo, HloOpcode::kAnd
      // clang-format on
      // TF_ASSIGN_OR_RETURN(
      //     HloInstruction * assertion,
      //     MakeCompareHlo(Comparison::Direction::kEq, dim1, dim2));
      // if (shape_assertion_ == nullptr) {
      //   shape_assertion_ = assertion;
      // } else {
      //   TF_ASSIGN_OR_RETURN(
      //       shape_assertion_,
      //       MakeBinaryHlo(HloOpcode::kAnd, shape_assertion_, assertion));
      // }
      // return absl::OkStatus();
      return absl::UnimplementedError(
          "DynamicDimensionInferenceVisitor::InsertShapeCheck not implemented "
          "when shape_check_mode_ is DynamicDimensionInference::kRuntime");
    }
    default:
      LOG(FATAL) << "Unreachable";
  }
}

absl::Status DynamicDimensionInferenceVisitor::ForEachDynamicDimensionInOperand(
    HloInstruction* inst, int64_t operand_index, OperandDynamicDimensionFn fn) {
  auto iter =
      parent_->per_hlo_dynamic_dimensions_.find(inst->operand(operand_index));
  if (iter != parent_->per_hlo_dynamic_dimensions_.end()) {
    for (auto& dynamic_dimension : iter->second) {
      HloInstruction* dynamic_size = parent_->GetDynamicSize(
          dynamic_dimension.inst, dynamic_dimension.index,
          dynamic_dimension.dim);
      TF_RETURN_IF_ERROR(fn(dynamic_dimension.inst, dynamic_dimension.index,
                            dynamic_dimension.dim, operand_index,
                            dynamic_size));
    }
  }
  return absl::OkStatus();
}

absl::Status DynamicDimensionInferenceVisitor::ForEachOperandDynamicDimension(
    HloInstruction* inst, OperandDynamicDimensionFn fn) {
  for (int64_t operand_index = 0; operand_index < inst->operand_count();
       ++operand_index) {
    TF_RETURN_IF_ERROR(
        ForEachDynamicDimensionInOperand(inst, operand_index, fn));
  }
  return absl::OkStatus();
}

void DynamicDimensionInference::SetDynamicSize(HloInstruction* inst,
                                               const ShapeIndex& index,
                                               int64_t dim,
                                               HloInstruction* size) {
  CHECK_NE(inst, nullptr);
  CHECK_NE(size, nullptr);
  VLOG(1) << "Set dimension inst " << inst->ToString() << " index "
          << index.ToString() << "@" << dim << " to " << size->ToShortString();
  const Shape& subshape = ShapeUtil::GetSubshape(inst->shape(), index);
  CHECK(!subshape.IsTuple()) << "Can't set a tuple shape to dynamic dimension";
  CHECK(dim < subshape.rank() && dim >= 0)
      << "Asked to set invalid dynamic dimension. Shape: "
      << subshape.ToString() << ", Dimension: " << dim;
  DynamicDimension dynamic_dimension{inst, index, dim};
  // If we have already set the dynamic size, it should be the same.
  auto [it, inserted] = dynamic_mapping_.try_emplace(dynamic_dimension, size);
  if (!inserted) {
    CHECK_EQ(size, it->second) << "old: " << it->second->ToShortString()
                               << ", new: " << size->ToShortString();
  }
  auto iter = per_hlo_dynamic_dimensions_.try_emplace(inst);
  iter.first->second.emplace(dynamic_dimension);
}

void DynamicDimensionInference::CopyMapping(
    HloInstruction* from, HloInstruction* to,
    const absl::flat_hash_map<HloInstruction*, HloInstruction*>*
        dynamic_size_map) {
  auto iter = per_hlo_dynamic_dimensions_.find(from);
  if (iter != per_hlo_dynamic_dimensions_.end()) {
    for (auto& dynamic_dimension : iter->second) {
      HloInstruction* dynamic_size =
          GetDynamicSize(dynamic_dimension.inst, dynamic_dimension.index,
                         dynamic_dimension.dim);
      if (dynamic_size_map != nullptr) {
        dynamic_size = dynamic_size_map->at(dynamic_size);
      }
      SetDynamicSize(to, dynamic_dimension.index, dynamic_dimension.dim,
                     dynamic_size);
    }
  }
}

// static
absl::StatusOr<DynamicDimensionInference> DynamicDimensionInference::Run(
    HloModule* module, OpSupportsDynamismHandler op_supports_dynamism_handler,
    CustomCallInferenceHandler custom_call_handler,
    ShapeCheckMode shape_check_mode,
    const AssertionGenerator& assertion_generator,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  DynamicDimensionInference inference(
      module, std::move(op_supports_dynamism_handler),
      std::move(custom_call_handler), shape_check_mode, assertion_generator,
      execution_threads);
  TF_RETURN_IF_ERROR(inference.AnalyzeDynamicDimensions());
  return std::move(inference);
}

std::string DynamicDimensionInference::ToString() const {
  std::vector<std::string> pieces;
  pieces.push_back("DynamicDimensionInference: ");
  for (const auto& mapping : dynamic_mapping_) {
    const DynamicDimension& dynamic_dimension = mapping.first;
    pieces.push_back(absl::StrFormat(
        " -- instruction %s at %s has dim %lld as dynamic"
        " dimension, which is represented by instruction %s",
        dynamic_dimension.inst->ToString(), dynamic_dimension.index.ToString(),
        dynamic_dimension.dim, mapping.second->ToString()));
  }
  return absl::StrJoin(pieces, "\n");
}

DynamicDimensionInference::DynamicDimensionInference(
    HloModule* module, OpSupportsDynamismHandler op_supports_dynamism_handler,
    CustomCallInferenceHandler custom_call_handler,
    ShapeCheckMode shape_check_mode, AssertionGenerator assertion_generator,
    const absl::flat_hash_set<std::string_view>& execution_threads)
    : module_(module),
      op_supports_dynamism_handler_(std::move(op_supports_dynamism_handler)),
      custom_call_handler_(std::move(custom_call_handler)),
      shape_check_mode_(shape_check_mode),
      assertion_generator_(assertion_generator),
      execution_threads_(execution_threads) {}

absl::Status DynamicDimensionInference::AnalyzeDynamicDimensions() {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloDataflowAnalysis> dataflow_analysis,
      HloDataflowAnalysis::Run(*module_, /*ssa_form=*/false,
                               /*bitcast_defines_value=*/true,
                               /*can_share_buffer=*/nullptr,
                               /*forwards_value=*/nullptr, execution_threads_));
  for (HloComputation* computation : module_->MakeComputationPostOrder()) {
    if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                          execution_threads_)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(
        bool changed,
        DynamicDimensionInferenceVisitor::Run(
            computation, *dataflow_analysis, {}, this, custom_call_handler_,
            shape_check_mode_, assertion_generator_));
    changed_ |= changed;
  }
  return absl::OkStatus();
}

void DynamicDimensionInference::ReplaceAllDynamicDimensionUsesWith(
    HloInstruction* replace, HloInstruction* with) {
  CHECK(Shape::Equal().IgnoreLayout()(replace->shape(),
                                      ShapeUtil::MakeScalarShape(S32)));
  CHECK(Shape::Equal().IgnoreLayout()(with->shape(),
                                      ShapeUtil::MakeScalarShape(S32)));
  for (auto& kv : dynamic_mapping_) {
    if (kv.second == replace) {
      kv.second = with;
    }
  }
}

absl::Status DynamicDimensionInference::ForwardDynamicSize(
    HloInstruction* inst, HloInstruction* new_inst, const ShapeIndex& index) {
  TF_RET_CHECK(ShapeUtil::Compatible(inst->shape(), new_inst->shape()));

  for (int64_t dim = 0; dim < inst->shape().rank(); ++dim) {
    DynamicDimension dynamic_dimension_new{new_inst, index, dim};
    DynamicDimension dynamic_dimension{inst, index, dim};
    auto iter = dynamic_mapping_.find(dynamic_dimension);
    if (iter != dynamic_mapping_.end()) {
      dynamic_mapping_.insert({dynamic_dimension_new, iter->second});
      auto iter = per_hlo_dynamic_dimensions_.try_emplace(new_inst);
      iter.first->second.emplace(dynamic_dimension_new);
    }
  }

  return absl::OkStatus();
}

bool DynamicDimensionInference::HasDynamicDimension(
    HloInstruction* inst, ShapeIndexView index) const {
  bool has_dynamic_dim = false;
  ShapeUtil::ForEachSubshape(inst->shape(), [&](const Shape& subshape,
                                                const ShapeIndex& subindex) {
    if (subshape.IsTuple()) {
      return;
    }
    if (ShapeIndexView(subindex).subspan(0, index.size()) != index) {
      return;
    }
    for (int64_t i = 0; i < subshape.dimensions_size(); ++i) {
      HloInstruction* operand_dynamic_size = GetDynamicSize(inst, subindex, i);
      if (operand_dynamic_size != nullptr) {
        has_dynamic_dim = true;
      }
    }
  });
  return has_dynamic_dim;
}

Shape DynamicDimensionInference::GetDynamicShape(HloInstruction* inst) {
  Shape shape = inst->shape();
  ShapeUtil::ForEachMutableSubshape(
      &shape, [&](Shape* subshape, const ShapeIndex& index) {
        if (!subshape->IsArray()) {
          return;
        }
        for (int64_t dimension = 0; dimension < subshape->rank(); ++dimension) {
          if (GetDynamicSize(inst, index, dimension) != nullptr) {
            subshape->set_dynamic_dimension(dimension, true);
          }
        }
      });

  return shape;
}

HloInstruction* DynamicDimensionInference::GetDynamicSize(
    HloInstruction* inst, const ShapeIndex& index, int64_t dim) const {
  auto iter = dynamic_mapping_.find(DynamicDimension{inst, index, dim});
  if (iter != dynamic_mapping_.end()) {
    return iter->second;
  }
  return nullptr;
}

const HloInstruction* DynamicDimensionInference::GetDynamicSize(
    const HloInstruction* inst, const ShapeIndex& index, int64_t dim) const {
  return GetDynamicSize(const_cast<HloInstruction*>(inst), index, dim);
}

std::vector<HloInstruction*> DynamicDimensionInference::GetDynamicSizes(
    HloInstruction* inst, const ShapeIndex& index) const {
  CHECK(ShapeUtil::IndexIsValid(inst->shape(), index));
  const int64_t rank = ShapeUtil::GetSubshape(inst->shape(), index).rank();
  std::vector<HloInstruction*> result(rank, nullptr);
  for (int64_t i = 0; i < rank; ++i) {
    result[i] = GetDynamicSize(inst, index, i);
  }
  return result;
}

bool DynamicDimensionInference::CanInfer(HloInstruction* hlo) {
  // If the result shape is static, there are no dynamic dimensions to infer.
  // However, if there are called computations, we may need to run inference on
  // them. Similarly, custom calls can do anything based on the user callbacks.
  if (hlo->shape().is_static() && hlo->called_computations().empty() &&
      hlo->opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  // The dimensions of all operands must either be 1) not dynamic, or 2) have a
  // recorded dynamic size. The only case where a dimension can be dynamic, but
  // where we have recorded a dynamic size is for SetDynamicSize instructions.
  bool ok = true;
  for (int64_t operand_index = 0; operand_index < hlo->operand_count();
       ++operand_index) {
    ShapeUtil::ForEachSubshape(
        hlo->operand(operand_index)->shape(),
        [&](const Shape& subshape, const ShapeIndex& shape_index) {
          if (!subshape.IsArray()) {
            return;
          }
          for (int64_t dimension = 0; dimension < subshape.rank();
               ++dimension) {
            bool shape_is_dynamic = subshape.is_dynamic_dimension(dimension);
            bool dynamic_size_recorded =
                GetDynamicSize(hlo->operand(operand_index), shape_index,
                               dimension) != nullptr;
            if (shape_is_dynamic && !dynamic_size_recorded) {
              VLOG(2) << "cannot infer " << hlo->ToShortString()
                      << " because operand " << operand_index << " ("
                      << hlo->operand(operand_index)->ToShortString() << ")"
                      << " subshape " << shape_index.ToString()
                      << " is missing dynamic size for dimension " << dimension;
              ok = false;
            }
            // Sanity check that we have cleared the dynamic dimension on the
            // shape if we have recorded the dynamic size.
            CHECK(hlo->operand(operand_index)->opcode() ==
                      HloOpcode::kSetDimensionSize ||
                  hlo->operand(operand_index)->opcode() ==
                      HloOpcode::kCustomCall ||
                  !shape_is_dynamic || !dynamic_size_recorded);
          }
        });
  }
  return ok;
}

}  // namespace zkx
