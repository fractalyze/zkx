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
#include "zkx/hlo/evaluator/hlo_evaluator.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/internal/endian.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/base/logging.h"
#include "zkx/hlo/evaluator/hlo_evaluator_typed_visitor.h"
#include "zkx/hlo/ir/hlo_clone_context.h"
#include "zkx/hlo/utils/hlo_query.h"
#include "zkx/layout.h"
#include "zkx/layout_util.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"
#include "zkx/service/compilation_environments.h"
#include "zkx/service/hlo_module_config.h"
#include "zkx/service/logical_buffer.h"
#include "zkx/service/pattern_matcher.h"
#include "zkx/service/shape_inference.h"

namespace zkx {
namespace {

using primitive_util::NativeTypeOf;

template <typename OperandT>
absl::StatusOr<Literal> Compare(const Shape& shape, Comparison comparison,
                                LiteralSlice lhs_literal,
                                LiteralSlice rhs_literal) {
  auto populate = [&](auto compare_op) -> absl::StatusOr<Literal> {
    Literal result(shape);
    TF_RETURN_IF_ERROR(result.PopulateParallel<bool>(
        [&](absl::Span<const int64_t> multi_index, int /*thread_id*/) {
          auto lhs = lhs_literal.Get<OperandT>(multi_index);
          auto rhs = rhs_literal.Get<OperandT>(multi_index);
          return compare_op(lhs, rhs);
        }));
    return std::move(result);
  };
  switch (comparison.GetDirection()) {
    case ComparisonDirection::kEq:
      return populate([](auto lhs, auto rhs) { return lhs == rhs; });
    case ComparisonDirection::kNe:
      return populate([](auto lhs, auto rhs) { return lhs != rhs; });
    case ComparisonDirection::kGe:
      if constexpr (!math::IsEcPoint<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs >= rhs; });
      }
      break;
    case ComparisonDirection::kGt:
      if constexpr (!math::IsEcPoint<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs > rhs; });
        break;
      }
    case ComparisonDirection::kLe:
      if constexpr (!math::IsEcPoint<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs <= rhs; });
        break;
      }
    case ComparisonDirection::kLt:
      if constexpr (!math::IsEcPoint<OperandT>) {
        return populate([](auto lhs, auto rhs) { return lhs < rhs; });
        break;
      }
  }

  LOG(FATAL) << "unhandled direction for conversion to Comparison: "
             << comparison.ToString();
}

std::optional<bool> GetInstructionStaticValueAsBool(
    const HloInstruction* instruction) {
  HloEvaluator evaluator;
  absl::StatusOr<Literal> static_value =
      evaluator.Evaluate(instruction, /*precomputed_analyses=*/{},
                         /*recursively_evaluate_nonconstant_operands=*/true);
  if (static_value.ok()) {
    return static_value->GetFirstElement<bool>();
  }
  return std::nullopt;
}

template <PrimitiveType kType>
struct PopulateParallelImpl {
  using NativeT = NativeTypeOf<kType>;
  static absl::Status Run(
      Literal& literal,
      absl::FunctionRef<Literal(absl::Span<const int64_t>, int)>
          literal_generator) {
    return literal.PopulateParallel<NativeT>(
        [&literal_generator](absl::Span<const int64_t> output_index,
                             int thread_id) {
          return literal_generator(output_index, thread_id)
              .template Get<NativeT>({});
        });
  }
};

template <PrimitiveType kType>
struct PopulateImpl {
  using NativeT = NativeTypeOf<kType>;
  static absl::Status Run(
      Literal& literal,
      absl::FunctionRef<Literal(absl::Span<const int64_t>)> literal_generator) {
    return literal.Populate<NativeT>(
        [&literal_generator](absl::Span<const int64_t> output_index) {
          return literal_generator(output_index).template Get<NativeT>({});
        });
  }
};

// Helper function for when it has a Literal generator (typically from an
// embedded evaluator to evaluate subcomputations) but needs to extract the
// scalar value of a specific type from it to populate a Literal. Putting such
// evaluation implementations in typed visitors gives no performance benefits
// but leads to unnecessarily large code size therefore providing a delegation
// to small templated helpers just for the parts that require manipulating the
// native types to avoid templating the whole implementations.
template <template <PrimitiveType> typename Trait, typename F>
absl::Status Apply(Literal& literal, F&& literal_generator) {
  return primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&, literal_generator = std::forward<F>(literal_generator)](
          auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          return Trait<primitive_type_constant>::Run(
              literal, std::move(literal_generator));
        }
        LOG(FATAL) << "Unhandled primitive type "
                   << literal.shape().element_type();
      },
      literal.shape().element_type());
}

absl::Status MakeEvalErrorDueToParamOrInfeed(
    const HloInstruction& eval_instruction) {
  absl::Status error = absl::FailedPreconditionError(absl::StrCat(
      "Failed to evaluate instruction (", eval_instruction.name(),
      ") since it depends on infeed or parameters to its parent computation (",
      eval_instruction.parent()->name(), ")."));
  std::string error_payload;
  error_payload.resize(sizeof(internal::EvalErrorDetail));
  absl::little_endian::Store32(
      const_cast<char*>(error_payload.data()),
      static_cast<uint32_t>(
          internal::EvalErrorDetail::kDynamicValueDependence));
  error.SetPayload(internal::kEvalErrorDetailUrl, absl::Cord(error_payload));
  return error;
}

// Represents a value that might or might not be determined statically.
struct DynamicOrStaticInteger {
  std::optional<int64_t> static_value;
  bool is_dynamic() const { return !static_value.has_value(); }

  std::string ToString() const {
    return is_dynamic() ? std::string("DYNAMIC") : absl::StrCat(*static_value);
  }
};

std::optional<DynamicOrStaticInteger> GetInstructionValueAsInteger(
    const HloInstruction* instruction,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  HloEvaluator evaluator;
  absl::StatusOr<Literal> static_value =
      evaluator.Evaluate(instruction, precomputed_analyses,
                         /*recursively_evaluate_nonconstant_operands=*/true);
  if (static_value.ok()) {
    if (instruction->shape().element_type() == PrimitiveType::PRED) {
      return DynamicOrStaticInteger{
          static_cast<int64_t>(static_value->GetFirstElement<bool>())};
    } else {
      return DynamicOrStaticInteger{static_value->GetFirstInteger()};
    }
  }

  std::optional<internal::EvalErrorDetail> eval_error_detail =
      internal::ParseEvalErrorDetail(static_value.status());
  if (eval_error_detail.has_value() &&
      *eval_error_detail ==
          internal::EvalErrorDetail::kDynamicValueDependence) {
    return DynamicOrStaticInteger{std::nullopt};
  }
  return std::nullopt;
}

// Represents an index into the while argument tuple and / or a value.
// At least one of param_index and value has a value; both of them could have
// a value.
struct ParamIndexAndValue {
  std::optional<int64_t> param_index;
  std::optional<DynamicOrStaticInteger> value;

  bool IsValid() const { return param_index.has_value() || value.has_value(); }
  std::string ToString() const {
    return absl::StrCat(
        "param_index:",
        !param_index.has_value() ? std::string("UNKNOWN")
                                 : absl::StrCat(*param_index),
        ",", "value:",
        !value.has_value() ? std::string("UNKNOWN") : value->ToString());
  }
};

std::optional<ParamIndexAndValue> TryParsingInstructionAsParameterAndInteger(
    const HloInstruction* instruction,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  // Skip copies.
  if (instruction->opcode() == HloOpcode::kCopy) {
    return TryParsingInstructionAsParameterAndInteger(instruction->operand(0),
                                                      precomputed_analyses);
  }
  if (instruction->opcode() == HloOpcode::kCopyDone) {
    return TryParsingInstructionAsParameterAndInteger(
        instruction->operand(0)->operand(1), precomputed_analyses);
  }
  ParamIndexAndValue result;
  if (Match(instruction, match::GetTupleElement().WithOperand(
                             0, match::Parameter().WithParameterNum(0)))) {
    result.param_index = instruction->tuple_index();
  }
  std::optional<DynamicOrStaticInteger> integer_value =
      GetInstructionValueAsInteger(instruction, precomputed_analyses);
  result.value = std::move(integer_value);
  if (!result.IsValid()) {
    return std::nullopt;
  }
  return std::optional<ParamIndexAndValue>(std::move(result));
}

// Represents the while loop condition comparison.
// We assume comparison is of the form: lhs comp rhs.
struct WhileCondComparison {
  ComparisonDirection comparison_direction;
  ParamIndexAndValue lhs;
  ParamIndexAndValue rhs;

  std::string ToString() const {
    return absl::StrCat("WhileCondComparison{", "LHS:{", lhs.ToString(),
                        "},RHS:{", rhs.ToString(), "}}");
  }
};

// Represents the parsed while loop condition. The loop induction variable may
// either be used in a comparison or returned directly, i.e., NoOp. In the case
// of NoOp, it contains the parameter index and initial value of the loop
// induction variable.
using WhileCondComparisonOrNoOp =
    std::variant<WhileCondComparison, ParamIndexAndValue>;

std::optional<ParamIndexAndValue> ParseComparisonOperand(
    const HloInstruction* operand,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (operand->opcode() == HloOpcode::kCopy ||
      operand->opcode() == HloOpcode::kCopyStart ||
      operand->opcode() == HloOpcode::kCopyDone) {
    return ParseComparisonOperand(operand->operand(0), precomputed_analyses);
  }
  std::optional<int64_t> param_index;
  if (Match(operand, match::GetTupleElement().WithOperand(
                         0, match::Parameter().WithParameterNum(0)))) {
    param_index = operand->tuple_index();
  }
  std::optional<DynamicOrStaticInteger> operand_value =
      GetInstructionValueAsInteger(operand, precomputed_analyses);
  if (!param_index.has_value() && !operand_value.has_value()) {
    return std::nullopt;
  }
  return ParamIndexAndValue{param_index, operand_value};
}

std::optional<WhileCondComparisonOrNoOp> PatternMatchLoopCondComparison(
    const HloInstruction* comparison,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  CHECK_EQ(comparison->opcode(), HloOpcode::kCompare);
  std::optional<ParamIndexAndValue> lhs =
      ParseComparisonOperand(comparison->operand(0), precomputed_analyses);
  std::optional<ParamIndexAndValue> rhs =
      ParseComparisonOperand(comparison->operand(1), precomputed_analyses);
  if (!lhs.has_value() || !rhs.has_value()) {
    return std::nullopt;
  }
  return WhileCondComparison{comparison->comparison_direction(),
                             *std::move(lhs), *std::move(rhs)};
}
// Finds the while loop condition comparison by matching the loop condition root
// with known patterns.
std::optional<WhileCondComparisonOrNoOp> PatternMatchLoopCondRoot(
    const HloInstruction* loop_cond_root,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (loop_cond_root->opcode() == HloOpcode::kCopy) {
    return PatternMatchLoopCondRoot(loop_cond_root->operand(0),
                                    precomputed_analyses);
  }
  if (loop_cond_root->opcode() == HloOpcode::kCopyDone) {
    return PatternMatchLoopCondRoot(loop_cond_root->operand(0)->operand(1),
                                    precomputed_analyses);
  }
  if (loop_cond_root->opcode() == HloOpcode::kCompare) {
    // Base pattern #1: gte-0 comp gte-1
    // Base pattern #2: constant comp gte
    // Base pattern #3: gte comp constant
    return PatternMatchLoopCondComparison(loop_cond_root, precomputed_analyses);
  }
  // Base pattern #4: gte is a boolean scalar and it was return immediately.
  if (Match(loop_cond_root, match::GetTupleElement().WithOperand(
                                0, match::Parameter().WithParameterNum(0)))) {
    if (loop_cond_root->shape().element_type() != PrimitiveType::PRED &&
        loop_cond_root->shape().rank() != 0) {
      return std::nullopt;
    }
    return ParamIndexAndValue{{/*param_index=*/loop_cond_root->tuple_index()}};
  }

  // Recursive pattern #1:
  // loop_cond_root is a GetTupleElement whose operand is a call with a single
  // parameter which takes the computation's single parameter.
  // In this case, if the called computation's root is a tuple, we can recurse
  // on that tuple's element as the new loop_cond_root.
  if (Match(loop_cond_root,
            match::GetTupleElement().WithOperand(
                0, match::Call().WithNumOperands(1).WithOperand(
                       0, match::Parameter().WithParameterNum(0))))) {
    const HloInstruction* call_instruction = loop_cond_root->operand(0);
    const HloComputation* to_apply = call_instruction->to_apply();
    const HloInstruction* to_apply_root = to_apply->root_instruction();
    if (Match(to_apply_root, match::Tuple())) {
      return PatternMatchLoopCondRoot(
          to_apply_root->operand(loop_cond_root->tuple_index()),
          precomputed_analyses);
    }
  }
  // Recursive pattern #2:
  // loop_cond_root is a GetTupleElement whose operand is a tuple.
  // We can recurse on the tuple's element as the new loop_cond_root.
  if (Match(loop_cond_root,
            match::GetTupleElement().WithOperand(0, match::Tuple()))) {
    const HloInstruction* new_cond_root =
        loop_cond_root->operand(0)->operand(loop_cond_root->tuple_index());
    return PatternMatchLoopCondRoot(new_cond_root, precomputed_analyses);
  }
  return std::nullopt;
}

std::optional<DynamicOrStaticInteger> PatternMatchInductionVarUpdate(
    const HloInstruction* induction_var_update, int64_t tuple_index,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (induction_var_update->opcode() == HloOpcode::kCopy) {
    return PatternMatchInductionVarUpdate(induction_var_update->operand(0),
                                          tuple_index, precomputed_analyses);
  }
  if (induction_var_update->opcode() == HloOpcode::kCopyDone) {
    return PatternMatchInductionVarUpdate(
        induction_var_update->operand(0)->operand(1), tuple_index,
        precomputed_analyses);
  }
  std::optional<ParamIndexAndValue> update_param_index_and_value =
      TryParsingInstructionAsParameterAndInteger(induction_var_update,
                                                 precomputed_analyses);

  if (update_param_index_and_value.has_value()) {
    if (update_param_index_and_value->param_index.has_value()) {
      if (*update_param_index_and_value->param_index == tuple_index) {
        // Pattern: the induc_var is directly returned from the loop body with
        // no changes.
        VLOG(3) << "PatternMatchInductionVarUpdate, pattern: [induc_var].";
        return DynamicOrStaticInteger{/*static_value=*/0};
      } else {
        VLOG(3)
            << "PatternMatchInductionVarUpdate, induction variable is set to "
               "another parameter value. Parsed update: "
            << update_param_index_and_value->ToString();
        return std::nullopt;
      }
    }
    if (update_param_index_and_value->value.has_value() &&
        !update_param_index_and_value->value->is_dynamic()) {
      VLOG(3) << "PatternMatchInductionVarUpdate, induction variable is set to "
                 "a constant. Parsed update: "
              << update_param_index_and_value->ToString();
      return std::nullopt;
    }
  }

  if (induction_var_update->opcode() != HloOpcode::kAdd &&
      induction_var_update->opcode() != HloOpcode::kSubtract) {
    return std::nullopt;
  }
  bool negate_update = induction_var_update->opcode() == HloOpcode::kSubtract;
  const HloInstruction* update_lhs = induction_var_update->operand(0);
  VLOG(3) << "PatternMatchInductionVarUpdate, LHS: " << update_lhs->ToString();
  std::optional<ParamIndexAndValue> update_lhs_param_index_and_value =
      TryParsingInstructionAsParameterAndInteger(update_lhs,
                                                 precomputed_analyses);

  const HloInstruction* update_rhs = induction_var_update->operand(1);
  VLOG(3) << "PatternMatchInductionVarUpdate, RHS: " << update_rhs->ToString();
  std::optional<ParamIndexAndValue> update_rhs_param_index_and_value =
      TryParsingInstructionAsParameterAndInteger(update_rhs,
                                                 precomputed_analyses);

  if (!update_lhs_param_index_and_value.has_value() ||
      !update_lhs_param_index_and_value->value.has_value() ||
      !update_rhs_param_index_and_value.has_value() ||
      !update_rhs_param_index_and_value->value.has_value()) {
    VLOG(3) << "PatternMatchInductionVarUpdate, failed to parse operands. "
               "Induction var update instruction: "
            << induction_var_update->ToString();
    return std::nullopt;
  }

  VLOG(3) << "update_lhs: " << update_lhs->ToString();
  VLOG(3) << "update_rhs: " << update_rhs->ToString();

  if (update_lhs_param_index_and_value->param_index.has_value() &&
      *update_lhs_param_index_and_value->param_index == tuple_index &&
      update_lhs_param_index_and_value->value->is_dynamic()) {
    if (update_rhs_param_index_and_value->value->is_dynamic()) {
      return update_rhs_param_index_and_value->value;
    }
    int64_t update_value =
        *update_rhs_param_index_and_value->value->static_value;
    return negate_update
               ? DynamicOrStaticInteger{/*static_value=*/-update_value}
               : DynamicOrStaticInteger{/*static_value=*/update_value};
  }

  if (update_rhs_param_index_and_value->param_index.has_value() &&
      *update_rhs_param_index_and_value->param_index == tuple_index &&
      update_rhs_param_index_and_value->value->is_dynamic() && !negate_update) {
    return update_lhs_param_index_and_value->value;
  }
  VLOG(3) << "Failed to pattern match induction variable update.";
  return std::nullopt;
}

// Tries to parse the loop body to find how the induction variable is updated
// using pattern matching.
std::optional<DynamicOrStaticInteger>
PatternMatchInductionVarUpdateFromLoopBodyRoot(
    const HloInstruction* loop_body_root, int64_t tuple_index,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  if (loop_body_root->opcode() != HloOpcode::kTuple ||
      loop_body_root->operand_count() <= tuple_index) {
    return std::nullopt;
  }
  const HloInstruction* induction_var_update =
      loop_body_root->operand(tuple_index);
  return PatternMatchInductionVarUpdate(induction_var_update, tuple_index,
                                        precomputed_analyses);
}

std::optional<bool> PatternMatchLoopCondVarOverride(
    const HloInstruction* loop_body_root, int64_t tuple_index) {
  if (!Match(loop_body_root, match::Tuple()) ||
      loop_body_root->operand_count() <= tuple_index) {
    return std::nullopt;
  }
  const HloInstruction* cond_var_override =
      loop_body_root->operand(tuple_index);
  return GetInstructionStaticValueAsBool(cond_var_override);
}

// A convenience wrapper to compute the while loop's argument's init value at
// the given tuple_index. If the init value depends on parameters to the
// while loop's parent computation or infeed, we consider the init value
// dynamic.
std::optional<DynamicOrStaticInteger> EvaluateWhileLoopParamInitValue(
    const HloInstruction* param_instruction, int64_t tuple_index) {
  if (param_instruction->opcode() != HloOpcode::kTuple) {
    return std::nullopt;
  }
  const HloInstruction* element_instruction =
      param_instruction->operand(tuple_index);
  return GetInstructionValueAsInteger(element_instruction,
                                      /*precomputed_analyses=*/{});
}

}  // namespace

namespace internal {

constexpr std::string_view kEvalErrorDetailUrl = "EvalErrorDetailUrl";

std::optional<EvalErrorDetail> ParseEvalErrorDetail(const absl::Status& error) {
  auto error_detail = error.GetPayload(kEvalErrorDetailUrl);
  if (!error_detail.has_value() || error_detail->empty()) {
    return std::nullopt;
  }
  return static_cast<EvalErrorDetail>(
      absl::little_endian::Load32(error_detail->Flatten().data()));
}

}  // namespace internal

std::optional<ParsedWhileLoop> HandleNoopLoopCondition(
    const ParamIndexAndValue& parameter_index_and_value,
    const HloInstruction* while_operand, const HloComputation* while_body) {
  CHECK(parameter_index_and_value.param_index.has_value());
  int64_t loop_cond_var_index = *parameter_index_and_value.param_index;
  std::optional<DynamicOrStaticInteger> noop_value =
      EvaluateWhileLoopParamInitValue(while_operand, loop_cond_var_index);

  if (noop_value.has_value()) {
    if (noop_value->is_dynamic()) {
      return kParsedDynamicWhileLoop;
    } else if (*noop_value->static_value == 0) {
      return ParsedWhileLoop{
          ParsedStaticWhileLoop{/*trip_count=*/0,
                                /*induction_var_index=*/loop_cond_var_index,
                                /*induction_var_init_value=*/0,
                                /*step_size=*/0,
                                /*loop_bound=*/0}};
    }
    std::optional<bool> updated_loop_cond_var = PatternMatchLoopCondVarOverride(
        while_body->root_instruction(), loop_cond_var_index);
    if (updated_loop_cond_var.has_value()) {
      if (!*updated_loop_cond_var) {
        return ParsedWhileLoop{
            ParsedStaticWhileLoop{/*trip_count=*/1,
                                  /*induction_var_index=*/loop_cond_var_index,
                                  /*induction_var_init_value=*/0,
                                  /*step_size=*/1,
                                  /*loop_bound=*/1}};
      } else {
        // This is an infinite loop and we set trip_count to -1.
        return ParsedWhileLoop{
            ParsedStaticWhileLoop{/*trip_count=*/-1,
                                  /*induction_var_index=*/loop_cond_var_index,
                                  /*induction_var_init_value=*/0,
                                  /*step_size=*/0,
                                  /*loop_bound=*/1}};
      }
    }
  }
  return std::nullopt;
}

int64_t ComputeTripCountFromComparison(int64_t init, int64_t bound,
                                       int64_t update,
                                       bool comparison_with_equal) {
  if (comparison_with_equal && init > bound) {
    return 0;
  }
  if (!comparison_with_equal && init >= bound) {
    return 0;
  }
  int64_t distance = bound - init;
  int64_t trip_count = (distance + update - 1) / update;
  CHECK_GE(trip_count, 0);
  // Additional logic to deal with equal comparison.
  if (comparison_with_equal && (bound - init) % update == 0) {
    trip_count += 1;
  }
  return trip_count;
}

std::optional<ParsedWhileLoop> HandleStaticLoopComparison(
    int64_t lhs, int64_t rhs, Comparison::Direction comparison_direction) {
  if ((comparison_direction == Comparison::Direction::kLt && lhs < rhs) ||
      (comparison_direction == Comparison::Direction::kLe && lhs <= rhs) ||
      (comparison_direction == Comparison::Direction::kGt && lhs > rhs) ||
      (comparison_direction == Comparison::Direction::kGe && lhs >= rhs) ||
      (comparison_direction == Comparison::Direction::kEq && lhs == rhs) ||
      (comparison_direction == Comparison::Direction::kNe && lhs != rhs)) {
    // This is an infinite loop and we set trip_count to -1.
    // There is no induction variable.
    return ParsedWhileLoop{ParsedStaticWhileLoop{/*trip_count=*/-1,
                                                 /*induction_var_index=*/-1,
                                                 /*induction_var_init_value=*/0,
                                                 /*step_size=*/0,
                                                 /*loop_bound=*/1}};
  }
  return ParsedWhileLoop{ParsedStaticWhileLoop{/*trip_count=*/0,
                                               /*induction_var_index=*/-1,
                                               /*induction_var_init_value=*/0,
                                               /*step_size=*/0,
                                               /*loop_bound=*/0}};
}

std::optional<ParsedWhileLoop> PatternMatchParseWhileLoop(
    const HloInstruction* while_op,
    HloEvaluator::PrecomputedAnalyses precomputed_analyses) {
  VLOG(3) << "PatternMatchParseWhileLoop, while_op: " << while_op->name();
  const HloComputation* while_cond = while_op->while_condition();
  const HloComputation* while_body = while_op->while_body();
  const HloInstruction* while_operand = while_op->operand(0);
  // Try to parse the loop condition comparison.
  std::optional<WhileCondComparisonOrNoOp> loop_comparison_or_noop =
      PatternMatchLoopCondRoot(while_cond->root_instruction(),
                               precomputed_analyses);
  if (!loop_comparison_or_noop.has_value()) {
    return std::nullopt;
  }
  if (loop_comparison_or_noop->index() == 1) {
    return HandleNoopLoopCondition(
        std::get<ParamIndexAndValue>(*loop_comparison_or_noop), while_operand,
        while_body);
  }
  CHECK_EQ(loop_comparison_or_noop->index(), 0);
  WhileCondComparison loop_comparison =
      std::get<WhileCondComparison>(*loop_comparison_or_noop);
  CHECK(loop_comparison.lhs.IsValid() && loop_comparison.rhs.IsValid());

  // We can't handle the case when the while loop argument is not a Tuple
  // instruction.
  if (while_operand->opcode() != HloOpcode::kTuple) {
    return std::nullopt;
  }

  if (!loop_comparison.lhs.value.has_value() ||
      !loop_comparison.rhs.value.has_value()) {
    return std::nullopt;
  }

  // We have either successfully parsed the init value for both LHS and RHS
  // or have returned as failure.
  CHECK(loop_comparison.lhs.value.has_value());
  CHECK(loop_comparison.rhs.value.has_value());

  VLOG(3) << loop_comparison.ToString();

  // If both operands of the loop condition comparison have dynamic value, the
  // trip count might be dynamic or static. This is a case that our existing
  // patterns could not yet handle, so we return std::nullopt.
  if (loop_comparison.lhs.value->is_dynamic() &&
      loop_comparison.rhs.value->is_dynamic()) {
    VLOG(3) << "Both operands of the loop condition comparison are dynamic.";
    return std::nullopt;
  }
  // We would have returned if both operands are dynamic. So there is at most
  // one dynamic operand, which is potentially the loop induction variable.
  CHECK(!loop_comparison.lhs.value->is_dynamic() ||
        !loop_comparison.rhs.value->is_dynamic());

  if (!loop_comparison.lhs.value->is_dynamic() &&
      !loop_comparison.rhs.value->is_dynamic()) {
    int64_t lhs_value = *loop_comparison.lhs.value->static_value;
    int64_t rhs_value = *loop_comparison.rhs.value->static_value;
    Comparison::Direction comparison_direction =
        loop_comparison.comparison_direction;
    return HandleStaticLoopComparison(lhs_value, rhs_value,
                                      comparison_direction);
  }
  std::optional<DynamicOrStaticInteger> induction_var_init;
  std::optional<DynamicOrStaticInteger> induction_var_update;
  bool lhs_is_induction_var = true;
  if (loop_comparison.lhs.value->is_dynamic()) {
    if (loop_comparison.lhs.param_index.has_value()) {
      VLOG(3) << "Comparison LHS is induction variable.";
      induction_var_init = EvaluateWhileLoopParamInitValue(
          while_operand, *loop_comparison.lhs.param_index);
      induction_var_update = PatternMatchInductionVarUpdateFromLoopBodyRoot(
          while_body->root_instruction(), *loop_comparison.lhs.param_index,
          precomputed_analyses);
      lhs_is_induction_var = true;
    }
  } else {
    CHECK(loop_comparison.rhs.value->is_dynamic());
    if (loop_comparison.rhs.param_index.has_value()) {
      VLOG(3) << "Comparison RHS is induction variable.";
      induction_var_init = EvaluateWhileLoopParamInitValue(
          while_operand, *loop_comparison.rhs.param_index);
      induction_var_update = PatternMatchInductionVarUpdateFromLoopBodyRoot(
          while_body->root_instruction(), *loop_comparison.rhs.param_index,
          precomputed_analyses);
      lhs_is_induction_var = false;
    }
  }

  if (!induction_var_init.has_value() || !induction_var_update.has_value()) {
    return std::nullopt;
  }
  VLOG(3) << "induction_var_init: " << induction_var_init->ToString();
  VLOG(3) << "induction_var_update: " << induction_var_update->ToString();
  if (induction_var_init->is_dynamic() || induction_var_update->is_dynamic()) {
    return kParsedDynamicWhileLoop;
  }

  int64_t init_value = *induction_var_init->static_value;
  int64_t update_value = *induction_var_update->static_value;
  Comparison::Direction comparison_direction =
      loop_comparison.comparison_direction;
  ParsedWhileLoop parsed_static_while_loop = ParsedWhileLoop{
      ParsedStaticWhileLoop{/*trip_count=*/0,
                            // Unassigned.
                            /*induction_var_index=*/-1,
                            /*induction_var_init_value=*/init_value,
                            /*step_size=*/update_value,
                            // Unassigned.
                            /*loop_bound=*/-1}};
  // Lhs is the induction variable.
  if (lhs_is_induction_var) {
    CHECK(loop_comparison.rhs.value.has_value() &&
          !loop_comparison.rhs.value->is_dynamic());
    int64_t bound = *loop_comparison.rhs.value->static_value;
    parsed_static_while_loop.static_while_loop->induction_var_index =
        *loop_comparison.lhs.param_index;
    parsed_static_while_loop.static_while_loop->loop_bound = bound;
    if (update_value > 0 &&
        (comparison_direction == Comparison::Direction::kLt ||
         comparison_direction == Comparison::Direction::kLe)) {
      int64_t trip_count = ComputeTripCountFromComparison(
          init_value, bound, update_value,
          comparison_direction == Comparison::Direction::kLe);
      parsed_static_while_loop.static_while_loop->trip_count = trip_count;
      return parsed_static_while_loop;
    }
    if (update_value < 0 &&
        (comparison_direction == Comparison::Direction::kGt ||
         comparison_direction == Comparison::Direction::kGe)) {
      int64_t trip_count = ComputeTripCountFromComparison(
          bound, init_value, -update_value,
          comparison_direction == Comparison::Direction::kGe);
      parsed_static_while_loop.static_while_loop->trip_count = trip_count;
      return parsed_static_while_loop;
    }
    return std::nullopt;
  }
  // Rhs is the induction variable.
  CHECK(loop_comparison.lhs.value.has_value() &&
        !loop_comparison.lhs.value->is_dynamic());
  int64_t bound = *loop_comparison.lhs.value->static_value;
  parsed_static_while_loop.static_while_loop->induction_var_index =
      *loop_comparison.rhs.param_index;
  parsed_static_while_loop.static_while_loop->loop_bound = bound;
  if (update_value > 0 &&
      (comparison_direction == Comparison::Direction::kGt ||
       comparison_direction == Comparison::Direction::kGe)) {
    int64_t trip_count = ComputeTripCountFromComparison(
        init_value, bound, update_value,
        comparison_direction == Comparison::Direction::kGe);
    parsed_static_while_loop.static_while_loop->trip_count = trip_count;
    return parsed_static_while_loop;
  }
  if (update_value < 0 &&
      (comparison_direction == Comparison::Direction::kLt ||
       comparison_direction == Comparison::Direction::kLe)) {
    int64_t trip_count = ComputeTripCountFromComparison(
        bound, init_value, -update_value,
        comparison_direction == Comparison::Direction::kLe);
    parsed_static_while_loop.static_while_loop->trip_count = trip_count;
    return parsed_static_while_loop;
  }
  return std::nullopt;
}

// Note that unsupported types by the typed visitor does not necessarily imply
// the non-typed HloEvaluator (parent evaluator) would not support them either
// in the type-agnostic handler. For e.g., HandleGetTupleElement in the parent
// type-agnostic evaluator will be able to accept Tuple primitive type, whereas
// HloEvaluatorTypedVisitor cannot.
HloEvaluator::HloEvaluator(int64_t max_loop_iterations)
    : max_loop_iterations_(max_loop_iterations) {
  for (int i = PrimitiveType_MIN; i < PrimitiveType_ARRAYSIZE; ++i) {
    if (!primitive_util::IsArrayType(PrimitiveType{i})) {
      continue;
    }
    primitive_util::PrimitiveTypeSwitch<void>(
        [&](auto primitive_type) {
          if constexpr (primitive_util::IsArrayType(primitive_type)) {
            using NativeT = primitive_util::NativeTypeOf<primitive_type>;
            if constexpr (primitive_util::IsSignedIntegralType(
                              primitive_type)) {
              typed_visitors_[primitive_type] =
                  std::make_unique<HloEvaluatorTypedVisitor<NativeT, int64_t>>(
                      this);
            } else if constexpr (primitive_util::IsUnsignedIntegralType(
                                     primitive_type)) {
              typed_visitors_[primitive_type] =
                  std::make_unique<HloEvaluatorTypedVisitor<NativeT, uint64_t>>(
                      this);
            } else {
              typed_visitors_[primitive_type] =
                  std::make_unique<HloEvaluatorTypedVisitor<NativeT>>(this);
            }
          }
        },
        PrimitiveType{i});
  }

  typed_visitors_[TUPLE] =
      std::make_unique<ConstFunctionVisitor>([](const HloInstruction*) {
        return absl::UnimplementedError(
            "HloEvaluatorTypedVisitor: unhandled primitive type: TUPLE.");
      });
  typed_visitors_[OPAQUE_TYPE] =
      std::make_unique<ConstFunctionVisitor>([](const HloInstruction*) {
        return absl::UnimplementedError(
            "HloEvaluatorTypedVisitor: unhandled primitive type: OPAQUE_TYPE.");
      });
  typed_visitors_[TOKEN] =
      std::make_unique<ConstFunctionVisitor>([](const HloInstruction*) {
        return absl::UnimplementedError(
            "HloEvaluatorTypedVisitor: unhandled primitive type: TOKEN.");
      });
}

absl::StatusOr<Literal> HloEvaluator::Evaluate(
    const HloComputation& computation,
    absl::Span<const Literal* const> arg_literals) {
  CHECK(computation.parent() != nullptr);
  ZKX_VLOG_LINES(
      2, "HloEvaluator::Evaluate computation:\n" + computation.ToString());
  OnEvaluateComputation(computation);

  if (arg_literals.size() != computation.num_parameters()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected %d argument%s, but got %d.", computation.num_parameters(),
        computation.num_parameters() == 1 ? "" : "s", arg_literals.size()));
  }
  for (int64_t i = 0; i < arg_literals.size(); ++i) {
    const auto& computation_shape =
        computation.parameter_instruction(i)->shape();
    const auto& arg_shape = arg_literals[i]->shape();
    bool ignore_layout = !computation_shape.has_layout();
    if (!Shape::Equal()
             .IgnoreLayout(ignore_layout)
             .MinorToMajorOnlyInLayout()(computation_shape, arg_shape)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Shape mismatch at parameter %d. Computation expected %s, but arg "
          "was %s.",
          i, ShapeUtil::HumanStringWithLayout(computation_shape),
          ShapeUtil::HumanStringWithLayout(arg_shape)));
    }
  }

  evaluated_.clear();
  arg_literals_.clear();
  call_graph_cache_.reset();
  tuple_points_to_analysis_cache_.reset();
  for (const auto& literal_ptr : arg_literals) {
    arg_literals_.push_back(&*literal_ptr);
  }

  // Re-seed RNG, either from the configuration's seed or a monotonic
  // per-evaluator seed (which prevents two evaluators from returning the same
  // random sequence).
  if (computation.parent()->config().seed()) {
    seed_ = computation.parent()->config().seed();
  } else {
    // Start global_seed at a (true) random value.
    static std::atomic<uint64_t> global_seed{std::random_device()()};
    seed_ = global_seed.fetch_add(1);
  }
  engine_.seed(seed_);

  TF_RETURN_IF_ERROR(computation.Accept(this));
  const Literal& result =
      GetEvaluatedLiteralFor(computation.root_instruction());
  if (VLOG_IS_ON(100)) {
    for (const HloInstruction* instr : computation.instructions()) {
      VLOG(100) << instr->name() << " = " << GetEvaluatedLiteralFor(instr);
    }
  }
  if (!result.IsKnown()) {
    return MakeEvalErrorDueToParamOrInfeed(*computation.root_instruction());
  }
  return result.Clone();
}

absl::StatusOr<Literal> HloEvaluator::Evaluate(
    const HloInstruction* instruction, PrecomputedAnalyses precomputed_analyses,
    bool recursively_evaluate_nonconstant_operands) {
  arg_literals_.clear();
  evaluated_.clear();
  call_graph_cache_.reset();
  tuple_points_to_analysis_cache_.reset();
  auto enable_partial_evaluation_cleanup =
      absl::MakeCleanup([this] { enable_partial_evaluation_ = false; });
  enable_partial_evaluation_ = recursively_evaluate_nonconstant_operands;
  TF_RETURN_IF_ERROR(
      EvaluateInternal(instruction, precomputed_analyses, /*shape_index=*/{},
                       recursively_evaluate_nonconstant_operands));
  const Literal& result = GetEvaluatedLiteralFor(instruction);
  if (!result.IsKnown()) {
    return MakeEvalErrorDueToParamOrInfeed(*instruction);
  }
  return result.Clone();
}

bool HloEvaluator::TryEvaluate(const HloInstruction* instruction,
                               Literal* result,
                               bool recursively_evaluate_nonconstant_operands) {
  CHECK_NE(result, nullptr);
  auto result_or = Evaluate(instruction, /*precomputed_analyses=*/{},
                            recursively_evaluate_nonconstant_operands);
  if (!result_or.ok()) {
    VLOG(1) << "TryEvaluate failed:" << result_or.status();
    return false;
  }

  *result = std::move(result_or).value();
  return true;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateWithSubstitutions(
    const HloInstruction* instruction,
    const absl::flat_hash_map<const HloInstruction*, const LiteralBase*>&
        substitutions,
    bool recursively_evaluate_nonconstant_operands) {
  std::vector<std::unique_ptr<HloInstruction>> owned_operands;
  for (const HloInstruction* operand : instruction->operands()) {
    auto it = substitutions.find(operand);
    if (it == substitutions.end()) {
      if (recursively_evaluate_nonconstant_operands) {
        TF_ASSIGN_OR_RETURN(Literal value,
                            EvaluateWithSubstitutions(
                                operand, substitutions,
                                recursively_evaluate_nonconstant_operands));
        owned_operands.push_back(HloInstruction::CreateConstant(value.Clone()));
      } else {
        if (!operand->IsConstant()) {
          VLOG(2) << "EvaluateWithSubstitutions called when not all operands "
                     "are constant. Consider calling it with "
                     "`recursively_evaluate_non_constant_operands` true.";
        }
        owned_operands.push_back(operand->Clone());
      }
    } else {
      owned_operands.push_back(
          HloInstruction::CreateConstant(it->second->Clone()));
    }
  }

  std::vector<HloInstruction*> operands;
  operands.reserve(owned_operands.size());
  for (auto& operand : owned_operands) {
    operands.push_back(operand.get());
  }

  std::unique_ptr<HloInstruction> cloned_instruction =
      instruction->CloneWithNewOperands(instruction->shape(), operands);
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseBinaryOp(
    HloOpcode opcode, const Literal& lhs, const Literal& rhs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateBinary(lhs.shape(), opcode, lhs_instr.get(),
                                   rhs_instr.get());
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseTernaryOp(
    HloOpcode opcode, const Literal& lhs, const Literal& rhs,
    const Literal& ehs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());
  std::unique_ptr<HloInstruction> ehs_instr =
      HloInstruction::CreateConstant(ehs.Clone());
  TF_ASSIGN_OR_RETURN(auto output_shape,
                      ShapeInference::InferTernaryOpShape(
                          opcode, lhs.shape(), rhs.shape(), ehs.shape()));
  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateTernary(output_shape, opcode, lhs_instr.get(),
                                    rhs_instr.get(), ehs_instr.get());
  return Evaluate(cloned_instruction.get());
}

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseCompareOp(
    ComparisonDirection direction, const Literal& lhs, const Literal& rhs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateCompare(
          ShapeUtil::ChangeElementType(lhs.shape(), PRED), lhs_instr.get(),
          rhs_instr.get(), direction);
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateElementwiseUnaryOp(
    HloOpcode opcode, const Literal& operand) {
  std::unique_ptr<HloInstruction> operand_instr =
      HloInstruction::CreateConstant(operand.Clone());

  TF_ASSIGN_OR_RETURN(Shape inferred_shape, ShapeInference::InferUnaryOpShape(
                                                opcode, operand.shape()));
  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateUnary(inferred_shape, opcode, operand_instr.get());
  auto result = Evaluate(cloned_instruction.get());

  return result;
}

absl::StatusOr<Literal> HloEvaluator::EvaluateDotOp(
    const DotDimensionNumbers& dim_numbers, const Literal& lhs,
    const Literal& rhs) {
  std::unique_ptr<HloInstruction> lhs_instr =
      HloInstruction::CreateConstant(lhs.Clone());
  std::unique_ptr<HloInstruction> rhs_instr =
      HloInstruction::CreateConstant(rhs.Clone());

  TF_ASSIGN_OR_RETURN(
      Shape dot_shape,
      ShapeInference::InferDotOpShape(lhs.shape(), rhs.shape(), dim_numbers,
                                      /*preferred_element_type=*/std::nullopt));

  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateDot(dot_shape, lhs_instr.get(), rhs_instr.get(),
                                dim_numbers);
  return Evaluate(cloned_instruction.get());
}

absl::Status HloEvaluator::EvaluateParameterFromCallerArgument(
    const HloInstruction* parameter, const ShapeIndex& shape_index,
    PrecomputedAnalyses analyses) {
  CHECK(!evaluated_.contains(parameter));
  const HloComputation* parent_computation = parameter->parent();
  std::vector<HloInstruction*> computation_callers =
      analyses.call_graph->GetComputationCallers(parent_computation);
  // If the parent computation has multiple callers, we cannot determine from
  // which caller the arguments are passed.
  if (computation_callers.size() != 1) {
    return absl::FailedPreconditionError(
        absl::StrCat("The computation ", parent_computation->name(),
                     " is called by ", computation_callers.size(),
                     " callers and thus its argument value "
                     "cannot be determined statically."));
  }
  const HloInstruction* computation_caller = computation_callers[0];
  const HloInstruction* caller_operand = computation_caller->operand(0);
  if (computation_caller->opcode() != HloOpcode::kWhile &&
      computation_caller->opcode() != HloOpcode::kCall) {
    return absl::FailedPreconditionError(absl::StrCat(
        "The computation ", parent_computation->name(), " is called by ",
        "instruction ", computation_caller->name(),
        ", which is not yet supported."));
  }
  if (computation_caller->opcode() == HloOpcode::kWhile) {
    HloComputation* while_body = computation_caller->while_body();
    TF_ASSIGN_OR_RETURN(
        const LogicalBuffer* logical_buffer,
        analyses.tuple_points_to->GetBufferDefinedAt(
            while_body->parameter_instruction(parameter->parameter_number()),
            shape_index));
    const TuplePointsToAnalysis::BufferAliasVector& buffer_aliases =
        analyses.tuple_points_to->GetBufferAliases(*logical_buffer);
    bool unchanged_in_return = false;
    for (const BufferAlias& buffer_alias : buffer_aliases) {
      if (buffer_alias.instruction() == while_body->root_instruction() &&
          buffer_alias.index() == shape_index) {
        unchanged_in_return = true;
      }
    }
    if (!unchanged_in_return) {
      return MakeEvalErrorDueToParamOrInfeed(*parameter);
    }
  }
  TF_RETURN_IF_ERROR(
      EvaluateInternal(caller_operand, analyses, shape_index, true));
  const Literal& caller_operand_literal =
      GetEvaluatedLiteralFor(caller_operand);
  evaluated_[parameter] =
      Literal::CreateFromShapeWithUnknownLeafArrays(parameter->shape());
  TF_RETURN_IF_ERROR(evaluated_[parameter].CopyFrom(
      caller_operand_literal, /*dest_shape_index=*/shape_index,
      /*src_shape_index=*/shape_index));
  return absl::OkStatus();
}

std::vector<int64_t> HloEvaluator::GetS64Indices(
    absl::Span<HloInstruction* const> start_indices) {
  auto get_first_s64 = [&](const Literal& index) -> int64_t {
    return primitive_util::PrimitiveTypeSwitch<int64_t>(
        [&](auto primitive_type_constant) -> int64_t {
          if constexpr (primitive_util::IsIntegralType(
                            primitive_type_constant)) {
            return static_cast<int64_t>(
                index.GetFirstElement<NativeTypeOf<primitive_type_constant>>());
          }
          LOG(FATAL) << "GetS64Indices: unhandled primitive type for "
                     << PrimitiveType_Name(index.shape().element_type());
        },
        index.shape().element_type());
  };
  std::vector<int64_t> start;
  start.reserve(start_indices.size());
  for (HloInstruction* index : start_indices) {
    start.push_back(get_first_s64(GetEvaluatedLiteralFor(index)));
  }
  return start;
}

DimensionVector HloEvaluator::MakeDimMultipliers(const Shape& shape) {
  DimensionVector v(shape.rank());
  int64_t scale = 1;
  for (auto dim : LayoutUtil::MinorToMajor(shape)) {
    v[dim] = scale;
    scale *= shape.dimensions(dim);
  }
  return v;
}

absl::Status HloEvaluator::EvaluateInternal(
    const HloInstruction* instruction, PrecomputedAnalyses precomputed_analyses,
    const ShapeIndex& shape_index,
    bool recursively_evaluate_nonconstant_operands) {
  // Don't need to evaluate this instruction again if it has already been
  // evaluated.
  if (IsAlreadyEvaluated(instruction, shape_index)) {
    return absl::OkStatus();
  }

  if (!recursively_evaluate_nonconstant_operands) {
    if (!hlo_query::AllOperandsAreConstants(*instruction)) {
      return absl::FailedPreconditionError(
          absl::StrCat("Not all operands are constants. Instruction: ",
                       instruction->ToString()));
    }
  } else {
    if (instruction->opcode() == HloOpcode::kGetTupleElement) {
      ShapeIndex new_shape_index = shape_index;
      new_shape_index.push_front(instruction->tuple_index());
      TF_RETURN_IF_ERROR(EvaluateInternal(
          instruction->operand(0), precomputed_analyses, new_shape_index,
          /*recursively_evaluate_nonconstant_operands=*/true));
    } else if (instruction->opcode() == HloOpcode::kTuple &&
               !shape_index.empty()) {
      ShapeIndex new_shape_index = shape_index;
      int64_t tuple_index = new_shape_index.front();
      new_shape_index.pop_front();
      TF_RETURN_IF_ERROR(
          EvaluateInternal(instruction->operand(tuple_index),
                           precomputed_analyses, new_shape_index,
                           /*recursively_evaluate_nonconstant_operands=*/true));
    } else if (instruction->opcode() == HloOpcode::kParameter) {
      CallGraph* call_graph =
          (precomputed_analyses.call_graph != nullptr)
              ? precomputed_analyses.call_graph
              : std::invoke([this, instruction]() -> CallGraph* {
                  call_graph_cache_ =
                      CallGraph::Build(instruction->GetModule());
                  return call_graph_cache_.get();
                });
      TuplePointsToAnalysis* tuple_points_to_analysis =
          (precomputed_analyses.tuple_points_to != nullptr)
              ? precomputed_analyses.tuple_points_to
              : std::invoke([this, instruction]() -> TuplePointsToAnalysis* {
                  absl::StatusOr<std::unique_ptr<TuplePointsToAnalysis>>
                      tuple_points_to_analysis =
                          TuplePointsToAnalysis::Run(instruction->GetModule());
                  if (!tuple_points_to_analysis.ok()) {
                    return nullptr;
                  }
                  tuple_points_to_analysis_cache_ =
                      *std::move(tuple_points_to_analysis);
                  return tuple_points_to_analysis_cache_.get();
                });
      if (call_graph && tuple_points_to_analysis) {
        absl::Status argument_eval_status = EvaluateParameterFromCallerArgument(
            instruction, shape_index, {tuple_points_to_analysis, call_graph});
        if (!argument_eval_status.ok()) {
          VLOG(4) << "Failed to evaluate parameter " << instruction->name()
                  << " from caller. Reason: " << argument_eval_status.message();
        } else {
          VLOG(4) << "Successfully evaluated parameter: "
                  << instruction->name();
        }
      }
    } else {
      for (HloInstruction* operand : instruction->operands()) {
        TF_RETURN_IF_ERROR(EvaluateInternal(
            operand, precomputed_analyses, /*shape_index=*/{},
            /*recursively_evaluate_nonconstant_operands=*/true));
        // Except for the above and following cases, we do not support handling
        // unknown operands for other HLOs. So mark the result as unknown.
        if ((!GetEvaluatedLiteralFor(operand).IsKnown() &&
             instruction->opcode() != HloOpcode::kCopy &&
             instruction->opcode() != HloOpcode::kCopyStart &&
             instruction->opcode() != HloOpcode::kCopyDone &&
             instruction->opcode() != HloOpcode::kAsyncStart &&
             instruction->opcode() != HloOpcode::kAsyncUpdate &&
             instruction->opcode() != HloOpcode::kAsyncDone &&
             instruction->opcode() != HloOpcode::kWhile)) {
          evaluated_[instruction] =
              Literal::CreateFromShapeWithUnknownLeafArrays(
                  instruction->shape());
          return absl::OkStatus();
        }
      }
    }
  }
  visitor_shape_index_ = shape_index;
  TF_RETURN_IF_ERROR(Preprocess(instruction));
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  TF_RETURN_IF_ERROR(Postprocess(instruction));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleBitcast(const HloInstruction* bitcast) {
  Shape result_shape = bitcast->shape();

  // Allow effective scalars without layouts as the result is unambiguous.
  if (!result_shape.has_layout() &&
      ShapeUtil::IsEffectiveScalar(result_shape)) {
    result_shape = LayoutUtil::GetWithDefaultLayout(result_shape);
  }

  // In general, we require a layout to evaluate a bitcast: this is the only
  // operation where indexing is physical rather than logical.
  if (!result_shape.has_layout()) {
    return absl::InvalidArgumentError(
        "Evaluator cannot evaluate bitcast for non-scalar operand without "
        "assigned layout.");
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(result_shape));

  Literal result(result_shape);

  // Bitcast output is allowed to be smaller than the input if the backend-
  // specific buffer sizes for the input and output are the same. Since the HLO
  // evaluator doesn't have access to the backend-specific shape size function,
  // assume it's OK to bitcast if output <= input.
  const Literal& operand_literal = GetEvaluatedLiteralFor(bitcast->operand(0));
  TF_RET_CHECK(operand_literal.size_bytes() >= result.size_bytes());
  memcpy(result.untyped_data(), operand_literal.untyped_data(),
         result.size_bytes());
  evaluated_[bitcast] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleBitcastConvert(const HloInstruction* convert) {
  const HloInstruction* operand = convert->operand(0);
  TF_ASSIGN_OR_RETURN(
      Literal result,
      GetEvaluatedLiteralFor(operand).BitcastConvert(convert->shape()));

  evaluated_[convert] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleGetDimensionSize(
    const HloInstruction* get_dimension_size) {
  // TODO(chokobole): Uncomment this. Dependency: HloInstruction::dimension
  // const HloInstruction* operand = get_dimension_size->operand(0);
  // int64_t dim = get_dimension_size->dimension();
  // if (dynamic_dimension_inference_ == nullptr) {
  //   return absl::InvalidArgumentError(
  //       "Evaluator cannot evaluate get_dimension_size without "
  //       "set_dynamic_dimension_inference.");
  // }
  // const HloInstruction* dynamic_size =
  //     dynamic_dimension_inference_->GetDynamicSize(operand, {}, dim);
  // if (dynamic_size != nullptr) {
  //   evaluated_[get_dimension_size] =
  //       GetEvaluatedLiteralFor(dynamic_size).Clone();
  //   return absl::OkStatus();
  // }

  // const Shape& shape = get_dimension_size->operand(0)->shape();
  // Literal output(ShapeUtil::MakeShape(S32, {}));
  // output.PopulateWithValue(
  //     static_cast<int32_t>(shape.dimensions(get_dimension_size->dimension())));
  // evaluated_[get_dimension_size] = std::move(output);
  // return absl::OkStatus();
  return absl::UnimplementedError("HandleGetDimensionSize not implemented");
}

absl::Status HloEvaluator::HandleSetDimensionSize(
    const HloInstruction* set_dimension_size) {
  // TODO(chokobole): Uncomment this. Dependency: HloInstruction::dimension
  // const Literal& operand_literal =
  //     GetEvaluatedLiteralFor(set_dimension_size->operand(0));
  // Literal result(set_dimension_size->shape());
  // memcpy(result.untyped_data(), operand_literal.untyped_data(),
  //        operand_literal.size_bytes());
  // const Literal& size_literal =
  //     GetEvaluatedLiteralFor(set_dimension_size->operand(1));
  // result.SetDynamicSize(set_dimension_size->dimension(),
  //                       size_literal.Get<int32_t>({}));
  // evaluated_[set_dimension_size] = std::move(result);
  // return absl::OkStatus();
  return absl::UnimplementedError("HandleSetDimensionSize not implemented");
}

absl::Status HloEvaluator::HandleParameter(const HloInstruction* parameter) {
  if (!IsAlreadyEvaluated(parameter, visitor_shape_index_)) {
    if (!enable_partial_evaluation_) {
      return absl::FailedPreconditionError(
          "Failed to evaluate instruction since its operands are unknown "
          "or undetermined and partial evaluation is not enabled.");
    }
    evaluated_[parameter] =
        Literal::CreateFromShapeWithUnknownLeafArrays(parameter->shape());
    return absl::OkStatus();
  }

  if (!arg_literals_.empty()) {
    // Nothing to do other than sanity checks. Parameters' values are stored in
    // arg_literals_.
    CHECK_LT(parameter->parameter_number(), arg_literals_.size());
#ifndef NDEBUG
    const Literal* input_literal = arg_literals_[parameter->parameter_number()];
    VLOG(2) << "Parameter evaluated to: " << input_literal->ToString();
    bool check_layout = parameter->shape().has_layout();
    DCHECK(Shape::Equal()
               .IgnoreLayout(!check_layout)
               .MinorToMajorOnlyInLayout()(parameter->shape(),
                                           input_literal->shape()))
        << "parameter shape is: "
        << ShapeUtil::HumanStringWithLayout(parameter->shape())
        << ", but input literal shape is: "
        << ShapeUtil::HumanStringWithLayout(input_literal->shape());
#endif
  }

  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleInfeed(const HloInstruction* infeed) {
  if (!enable_partial_evaluation_) {
    return absl::FailedPreconditionError(
        "Failed to evaluate instruction since its operands are unknown "
        "or undetermined and partial evaluation is not enabled.");
  }
  evaluated_[infeed] =
      Literal::CreateFromShapeWithUnknownLeafArrays(infeed->shape());
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConstant(const HloInstruction*) {
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleReshape(const HloInstruction* reshape) {
  TF_ASSIGN_OR_RETURN(evaluated_[reshape],
                      GetEvaluatedLiteralFor(reshape->operand(0))
                          .Reshape(reshape->shape().dimensions()));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleTranspose(const HloInstruction* transpose) {
  evaluated_[transpose] = GetEvaluatedLiteralFor(transpose->operand(0))
                              .Transpose(transpose->dimensions());
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConcatenate(
    const HloInstruction* concatenate) {
  absl::Span<HloInstruction* const> operands(concatenate->operands());
  // The result concatenate dimension is going to be the sum of all
  // concatenate dimensions of the operands taking part of the operation.
  const Shape& reference_shape = operands[0]->shape();
  CHECK(reference_shape.IsArray());
  const int64_t rank = reference_shape.rank();
  const int64_t concat_dim = concatenate->dimensions()[0];
  CHECK_GE(concat_dim, 0);
  CHECK_LT(concat_dim, rank);

  DimensionVector concat_dimensions(reference_shape.dimensions().begin(),
                                    reference_shape.dimensions().end());

  for (int64_t i = 1; i < operands.size(); ++i) {
    const Shape& operand_shape = operands[i]->shape();
    CHECK(operand_shape.IsArray());
    // Accumulate the concat dimension from all tensors taking part to the
    // operation.
    concat_dimensions[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  auto result_literal = LiteralUtil::CreateFromDimensions(
      reference_shape.element_type(), concat_dimensions);
  DimensionVector source_indices(rank, 0);
  DimensionVector dest_indices(concat_dimensions.size(), 0);

  for (auto operand : operands) {
    const Shape& operand_shape = operand->shape();
    TF_RETURN_IF_ERROR(result_literal.CopySliceFrom(
        GetEvaluatedLiteralFor(operand), source_indices, dest_indices,
        operand_shape.dimensions()));
    dest_indices[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  evaluated_[concatenate] = std::move(result_literal);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCompare(const HloInstruction* compare) {
  ComparisonDirection direction = compare->comparison_direction();
  ComparisonOrder order = compare->comparison_order();
  auto lhs = compare->operand(0);
  auto rhs = compare->operand(1);
  DCHECK(ShapeUtil::SameDimensions(compare->shape(), rhs->shape()) &&
         ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

  TF_RET_CHECK(lhs->shape().element_type() == rhs->shape().element_type());
  auto element_type = lhs->shape().element_type();
  Comparison comparison(direction, element_type, order);

  const Literal& lhs_literal = GetEvaluatedLiteralFor(lhs);
  const Literal& rhs_literal = GetEvaluatedLiteralFor(rhs);
  // Note here we switch on the operand's type.
  return primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          TF_ASSIGN_OR_RETURN(evaluated_[compare],
                              Compare<NativeT>(compare->shape(), comparison,
                                               lhs_literal, rhs_literal));
          return absl::OkStatus();
        }
        LOG(FATAL) << "HandleCompare: unknown primitive type: "
                   << PrimitiveType_Name(element_type);
      },
      element_type);
}

absl::Status HloEvaluator::HandleTuple(const HloInstruction* tuple) {
  std::vector<const Literal*> operand_literals;
  std::vector<Literal> operand_literal_values;
  if (!visitor_shape_index_.empty()) {
    // We only need to evaluate tuple at visitor_shape_index_. The other
    // operands might not have been evaluated, so mark the other operands as
    // undetermined.
    int64_t tuple_index = visitor_shape_index_.front();
    operand_literal_values.resize(tuple->operand_count());
    for (int operand_index = 0; operand_index < tuple->operand_count();
         ++operand_index) {
      if (operand_index == tuple_index) {
        operand_literals.push_back(
            &GetEvaluatedLiteralFor(tuple->operand(operand_index)));
      } else {
        operand_literal_values[operand_index] =
            Literal::CreateFromShapeWithUndeterminedLeafArrays(
                ShapeUtil::GetSubshape(tuple->shape(), {operand_index}));
        operand_literals.push_back(&operand_literal_values[operand_index]);
      }
    }
  } else {
    for (auto operand : tuple->operands()) {
      operand_literals.push_back(&GetEvaluatedLiteralFor(operand));
    }
  }

  // Inline part of LiteralUtil::MakeTuple() that avoids creating the leaf
  // buffers; these buffers can be extremely large.
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(operand_literals.size());
  for (const auto* element : operand_literals) {
    element_shapes.push_back(&element->shape());
  }
  Literal new_result = Literal::CreateFromShapeWithUndeterminedLeafArrays(
      ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int i = 0, end = operand_literals.size(); i < end; ++i) {
    TF_RETURN_IF_ERROR(
        new_result.CopyFrom(*operand_literals[i], /*dest_shape_index=*/{i}));
  }

  if (evaluated_.contains(tuple)) {
    CHECK(new_result.IsDetermined(visitor_shape_index_));
    TF_RETURN_IF_ERROR(
        evaluated_[tuple].CopyFrom(std::move(new_result),
                                   /*dest_shape_index=*/visitor_shape_index_,
                                   /*src_shape_index=*/visitor_shape_index_));
  } else {
    evaluated_[tuple] = std::move(new_result);
  }
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleFft(const HloInstruction* fft) {
  return absl::UnimplementedError("Fft is not implemented.");
  // const Literal& input_literal = GetEvaluatedLiteralFor(fft->operand(0));
  // Literal output_literal = Literal::CreateFromShape(fft->shape());

  // FftTransform<complex128> transform(fft);
  // TF_RETURN_IF_ERROR(transform.ComputeFft(fft, input_literal,
  // &output_literal)); evaluated_[fft] = std::move(output_literal);

  // return absl::OkStatus();
}

absl::Status HloEvaluator::HandleGather(const HloInstruction* gather) {
  // TODO(chokobole): Implement this. Dependency: GatherDimensionNumbers
  return absl::UnimplementedError("HandleGather not implemented");
}

absl::Status HloEvaluator::HandleScatter(const HloInstruction* hlo) {
  // TODO(chokobole): Implement this. Dependency: HloScatterInstruction
  return absl::UnimplementedError("HandleScatter not implemented");
}

absl::Status HloEvaluator::HandleBroadcast(const HloInstruction* broadcast) {
  const Literal& operand = GetEvaluatedLiteralFor(broadcast->operand(0));
  TF_RET_CHECK(broadcast->shape().element_type() ==
               operand.shape().element_type())
      << " broadcast from a different data type is not supported";
  TF_RET_CHECK(broadcast->dimensions().size() == operand.shape().rank())
      << "broadcast dimensions is of size: " << broadcast->dimensions().size()
      << " and rank of operand_to_broadcast is: " << operand.shape().rank();
  // Checks that operand's dimensions are the same as the broadcast's
  // dimensions along the dimensions to be broadcasted.
  for (int64_t i = 0; i < broadcast->dimensions().size(); ++i) {
    auto operand_dim_size = operand.shape().dimensions(i);
    auto broadcast_dim_size =
        broadcast->shape().dimensions(broadcast->dimensions(i));
    TF_RET_CHECK(operand_dim_size == broadcast_dim_size) << absl::StreamFormat(
        "Operand dimension %d is broadcast to output dimension %d, but the "
        "sizes of these two dims do not match (%d vs %d): %s",
        i, broadcast->dimensions(i), operand_dim_size, broadcast_dim_size,
        broadcast->ToString());
  }

  TF_ASSIGN_OR_RETURN(
      evaluated_[broadcast],
      operand.Broadcast(broadcast->shape(), broadcast->dimensions()));

  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAfterAll(const HloInstruction* after_all) {
  evaluated_[after_all] = LiteralUtil::CreateToken();
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAddDependency(
    const HloInstruction* add_dependency) {
  // AddDependency just forwards its zero-th operand.
  evaluated_[add_dependency] =
      GetEvaluatedLiteralFor(add_dependency->operand(0)).Clone();
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleGetTupleElement(
    const HloInstruction* get_tuple_element) {
  // clang-format off
  // TODO(chokobole): Implement this. Dependency: ShapeInference::InferGetTupleElementShape
  // clang-format on
  return absl::UnimplementedError("HandleGetTupleElement not implemented");
}

absl::Status HloEvaluator::HandleCopy(const HloInstruction* copy) {
  // If only the element type is different, try converting the literal.
  if (copy->shape().element_type() !=
      copy->operand(0)->shape().element_type()) {
    TF_ASSIGN_OR_RETURN(Literal result,
                        GetEvaluatedLiteralFor(copy->operand(0))
                            .Convert(copy->shape().element_type()));
    TF_RET_CHECK(ShapeUtil::Compatible(copy->shape(), result.shape()));
    evaluated_[copy] = std::move(result);
  } else {
    TF_RET_CHECK(
        ShapeUtil::Compatible(copy->shape(), copy->operand(0)->shape()));
    evaluated_[copy] = GetEvaluatedLiteralFor(copy->operand(0)).Clone();
  }
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAsyncStart(const HloInstruction* async_start) {
  std::vector<const Literal*> arg_literals;
  arg_literals.reserve(async_start->operands().size());
  for (auto operand : async_start->operands()) {
    const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
    arg_literals.push_back(&arg_literal);
  }

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(
      Literal result,
      embedded_evaluator->Evaluate(*async_start->async_wrapped_computation(),
                                   arg_literals));

  evaluated_[async_start] = Literal(async_start->shape());
  // Copy the operand values to the index {0, i} of the output.
  for (int i = 0; i < arg_literals.size(); ++i) {
    TF_RETURN_IF_ERROR(evaluated_[async_start].CopyFrom(
        *arg_literals[i], /*dest_shape_index=*/{0, i},
        /*src_shape_index=*/{}));
  }
  // Move the output value to the index {1} of the output.
  TF_RETURN_IF_ERROR(evaluated_[async_start].MoveFrom(
      std::move(result), /*dest_shape_index=*/{1}));

  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAsyncUpdate(
    const HloInstruction* async_update) {
  const Literal& operand_tuple_literal =
      GetEvaluatedLiteralFor(async_update->operand(0));
  evaluated_[async_update] = Literal(async_update->shape());
  TF_RETURN_IF_ERROR(evaluated_[async_update].CopyFrom(operand_tuple_literal,
                                                       /*dest_shape_index=*/{},
                                                       /*src_shape_index=*/{}));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleAsyncDone(const HloInstruction* async_done) {
  const Literal& operand_tuple_literal =
      GetEvaluatedLiteralFor(async_done->operand(0));
  evaluated_[async_done] = Literal(async_done->shape());
  TF_RETURN_IF_ERROR(evaluated_[async_done].CopyFrom(operand_tuple_literal,
                                                     /*dest_shape_index=*/{},
                                                     /*src_shape_index=*/{1}));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCopyStart(const HloInstruction* copy_start) {
  if (copy_start->user_count() != 1 ||
      copy_start->users().at(0)->opcode() != HloOpcode::kCopyDone) {
    return absl::FailedPreconditionError(
        absl::StrCat("Cannot evaluate a kCopyStart that doesn't have a single "
                     "kCopyDone user. Instruction: ",
                     copy_start->ToString()));
  }

  // The context in index {2} is undefined, but since we can't represent
  // undefined values using a Literal, we just use 0. This should be safe though
  // since we ensure that the only user of a kCopyStart is a kCopyDone which
  // consumes the context. Also note that MakeTuple copies its arguments, so
  // this is memory-safe.
  const Literal context_literal = LiteralUtil::CreateR0<uint32_t>(0);
  evaluated_[copy_start] = LiteralUtil::MakeTuple(
      {&GetEvaluatedLiteralFor(copy_start->operand(0)),
       &GetEvaluatedLiteralFor(copy_start->operand(0)), &context_literal});
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCopyDone(const HloInstruction* copy_done) {
  const HloInstruction* operand = copy_done->operand(0);
  if (operand->opcode() != HloOpcode::kCopyStart) {
    return absl::FailedPreconditionError(
        absl::StrCat("Cannot evaluate a kCopyDone that doesn't have a "
                     "kCopyStart as operand. Instruction: ",
                     copy_done->ToString()));
  }

  const Literal& operand_tuple_literal = GetEvaluatedLiteralFor(operand);
  evaluated_[copy_done] =
      Literal(ShapeUtil::GetTupleElementShape(operand->shape(), /*index=*/0));
  TF_RETURN_IF_ERROR(evaluated_[copy_done].CopyFrom(operand_tuple_literal,
                                                    /*dest_shape_index=*/{},
                                                    /*src_shape_index=*/{0}));
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCall(const HloInstruction* call) {
  auto* computation = call->to_apply();
  auto operands = call->operands();

  std::vector<const Literal*> arg_literals;
  arg_literals.reserve(operands.size());
  for (auto operand : operands) {
    const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
    arg_literals.push_back(&arg_literal);
  }

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(Literal result,
                      embedded_evaluator->Evaluate(*computation, arg_literals));

  evaluated_[call] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleFusion(const HloInstruction* fusion) {
  HloModuleConfig config;
  // Attach cloned computation to an empty HLO module so the existing ones are
  // not modified.
  HloModule empty_hlo_module("EmptyModuleForFusion", config,
                             std::make_unique<CompilationEnvironments>(
                                 fusion->GetModule()->comp_envs()));
  HloCloneContext context(&empty_hlo_module);
  auto cloned_fused_computation =
      fusion->fused_instructions_computation()->Clone(
          /*suffix=*/"clone_with_layout", &context);
  for (auto* instruction : cloned_fused_computation->instructions()) {
    if (!LayoutUtil::HasLayout(instruction->shape())) {
      LayoutUtil::SetToDefaultLayout(instruction->mutable_shape());
    }
  }
  auto readded_computation =
      empty_hlo_module.AddEntryComputation(std::move(cloned_fused_computation));

  auto operands = fusion->operands();
  std::vector<const Literal*> arg_literals;
  arg_literals.reserve(operands.size());
  for (auto operand : operands) {
    const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
    arg_literals.push_back(&arg_literal);
  }

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(Literal result, embedded_evaluator->Evaluate(
                                          *readded_computation, arg_literals));

  evaluated_[fusion] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConditional(
    const HloInstruction* conditional) {
  const auto& branch_index_literal =
      GetEvaluatedLiteralFor(conditional->operand(0));
  int branch_index;
  if (conditional->operand(0)->shape().element_type() == PRED) {
    branch_index = branch_index_literal.Get<bool>({}) ? 0 : 1;
  } else {
    branch_index = branch_index_literal.Get<int32_t>({});
    if (branch_index < 0 || branch_index >= conditional->branch_count()) {
      branch_index = conditional->branch_count() - 1;
    }
  }
  const auto& branch_computation_arg =
      GetEvaluatedLiteralFor(conditional->operand(1 + branch_index));

  std::unique_ptr<HloEvaluator> embedded_evaluator =
      CreateEmbedded(max_loop_iterations_);
  embedded_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  TF_ASSIGN_OR_RETURN(Literal result,
                      embedded_evaluator->Evaluate(
                          *conditional->branch_computation(branch_index),
                          {&branch_computation_arg}));

  evaluated_[conditional] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleConvert(const HloInstruction* convert) {
  const HloInstruction* operand = convert->operand(0);
  TF_RET_CHECK(ShapeUtil::SameDimensions(operand->shape(), convert->shape()));
  TF_ASSIGN_OR_RETURN(Literal result, GetEvaluatedLiteralFor(operand).Convert(
                                          convert->shape().element_type()));
  evaluated_[convert] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleDynamicSlice(
    const HloInstruction* dynamic_slice) {
  // TODO(chokobole): Implement this. Dependency: HloDynamicSliceInstruction
  return absl::UnimplementedError("HandleDynamicSlice not implemented");
}

absl::Status HloEvaluator::HandleDynamicUpdateSlice(const HloInstruction* dus) {
  // clang-format off
  // TODO(chokobole): Implement this. Dependency: HloDynamicUpdateSliceInstruction
  // clang-format on
  return absl::UnimplementedError("HandleDynamicUpdateSlice not implemented");
}

absl::Status HloEvaluator::HandleSelect(const HloInstruction* select) {
  const auto& pred = GetEvaluatedLiteralFor(select->operand(0));
  const auto& on_true = GetEvaluatedLiteralFor(select->operand(1));
  const auto& on_false = GetEvaluatedLiteralFor(select->operand(2));

  // If predicate is of scalar type, no element-wise selection would be needed.
  if (ShapeUtil::IsScalar(pred.shape())) {
    if (pred.Get<bool>({})) {
      evaluated_[select] = on_true.Clone();
    } else {
      evaluated_[select] = on_false.Clone();
    }
    return absl::OkStatus();
  }

  return DefaultAction(select);
}

namespace {

absl::StatusOr<Literal> CreateScalarLiteral(int64_t value,
                                            PrimitiveType element_type) {
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<Literal> {
        if constexpr (primitive_util::IsIntegralType(primitive_type_constant)) {
          return LiteralUtil::CreateR0(
              static_cast<NativeTypeOf<primitive_type_constant>>(value));
        }
        return absl::InvalidArgumentError("Unsupported element type.");
      },
      element_type);
}

// Parses the while loop if it matches one of the known patterns. Returns the
// value of the loop induction variable after the loop execution if the loop is
// static.
absl::StatusOr<Literal> TryParseAndEvaluateWhileInductionVar(
    const HloInstruction* while_hlo) {
  std::optional<ParsedWhileLoop> parsed_while_loop =
      PatternMatchParseWhileLoop(while_hlo, /*precomputed_analyses=*/{});
  if (!parsed_while_loop.has_value() || parsed_while_loop->is_dynamic()) {
    return absl::FailedPreconditionError(
        "Cannot evaluate a while loop's induction variable since the loop "
        "does not match a known loop pattern or the loop is not static.");
  }
  int64_t induction_var_value =
      parsed_while_loop->static_while_loop->induction_var_init_value +
      parsed_while_loop->static_while_loop->trip_count *
          parsed_while_loop->static_while_loop->step_size;
  Shape result_shape = while_hlo->shape().tuple_shapes(
      parsed_while_loop->static_while_loop->induction_var_index);
  TF_ASSIGN_OR_RETURN(
      Literal result,
      CreateScalarLiteral(induction_var_value, result_shape.element_type()));
  std::vector<Literal*> while_result_element_ptrs;
  while_result_element_ptrs.reserve(while_hlo->shape().tuple_shapes_size());
  std::vector<Literal> while_result_elements(
      while_hlo->shape().tuple_shapes_size());
  for (int i = 0; i < while_hlo->shape().tuple_shapes_size(); ++i) {
    if (i == parsed_while_loop->static_while_loop->induction_var_index) {
      while_result_element_ptrs.push_back(&result);
    } else {
      const Shape& shape = while_hlo->shape().tuple_shapes(i);
      while_result_elements[i] =
          Literal::CreateFromShapeWithUnknownLeafArrays(shape);
      while_result_element_ptrs.push_back(&while_result_elements[i]);
    }
  }
  return LiteralUtil::MakeTuple(while_result_element_ptrs);
}

}  // namespace

absl::Status HloEvaluator::HandleWhile(const HloInstruction* while_hlo) {
  const HloComputation* cond_comp = while_hlo->while_condition();
  const HloComputation* body_comp = while_hlo->while_body();
  // Initialize the loop carried valued with the input to the While instruction.
  auto lcv = GetEvaluatedLiteralFor(while_hlo->operand(0)).Clone();
  if (!lcv.IsKnown()) {
    std::optional<ParsedWhileLoop> parsed_while_loop =
        PatternMatchParseWhileLoop(while_hlo,
                                   /*precomputed_analyses=*/{});
    evaluated_[while_hlo] =
        Literal::CreateFromShapeWithUnknownLeafArrays(while_hlo->shape());
    if (!parsed_while_loop.has_value() || parsed_while_loop->is_dynamic() ||
        visitor_shape_index_.size() != 1 ||
        parsed_while_loop->static_while_loop->induction_var_index !=
            visitor_shape_index_[0]) {
      return absl::OkStatus();
    }
    Shape induction_var_shape =
        ShapeUtil::GetSubshape(while_hlo->shape(), visitor_shape_index_);
    int64_t trip_count = parsed_while_loop->static_while_loop->trip_count;
    TF_ASSIGN_OR_RETURN(
        Literal induction_var_val,
        CreateScalarLiteral(trip_count, induction_var_shape.element_type()));
    TF_RETURN_IF_ERROR(evaluated_[while_hlo].CopyFrom(
        induction_var_val, /*dest_shape_index=*/visitor_shape_index_,
        /*src_shape_index=*/{}));
    return absl::OkStatus();
  }
  bool keep_going = true;
  int64_t iteration_count = 0;
  std::unique_ptr<HloEvaluator> cond_evaluator =
      CreateEmbedded(max_loop_iterations_);
  cond_evaluator->set_dynamic_dimension_inference(dynamic_dimension_inference_);
  std::unique_ptr<HloEvaluator> loop_body_evaluator =
      CreateEmbedded(max_loop_iterations_);
  loop_body_evaluator->set_dynamic_dimension_inference(
      dynamic_dimension_inference_);
  while (keep_going) {
    if (max_loop_iterations_ >= 0 && iteration_count++ > max_loop_iterations_) {
      absl::StatusOr<Literal> result =
          TryParseAndEvaluateWhileInductionVar(while_hlo);
      if (result.ok()) {
        lcv = std::move(result).value();
        break;
      } else {
        return absl::InvalidArgumentError(
            absl::StrFormat("Loop %s exceeded loop iteration limit (%d).",
                            while_hlo->name(), max_loop_iterations_));
      }
    }
    TF_ASSIGN_OR_RETURN(auto cond_val,
                        cond_evaluator->Evaluate(*cond_comp, {&lcv}));
    keep_going = cond_val.GetFirstElement<bool>();
    if (keep_going) {
      TF_ASSIGN_OR_RETURN(auto body_val,
                          loop_body_evaluator->Evaluate(*body_comp, {&lcv}));
      VLOG(3) << "Loop iteration result: " << body_val.ToString();
      lcv = std::move(body_val);
      cond_evaluator->ResetVisitStates();
      loop_body_evaluator->ResetVisitStates();
    }
  }
  evaluated_[while_hlo] = std::move(lcv);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleReverse(const HloInstruction* reverse) {
  // clang-format off
  // TODO(chokobole): Implement this. Dependency: ShapeInference::InferReverseShape
  // clang-format on
  return absl::UnimplementedError("HandleReverse not implemented");
}

absl::Status HloEvaluator::HandleSlice(const HloInstruction* slice) {
  // clang-format off
  // TODO(chokobole): Implement this. Dependency: ShapeInference::InferSliceShape
  // clang-format on
  return absl::UnimplementedError("HandleSlice not implemented");
}

absl::Status HloEvaluator::HandleReduce(const HloInstruction* hlo) {
  // TODO(chokobole): Implement this. Dependency: HloReduceInstruction
  return absl::UnimplementedError("HandleReduce not implemented");
}

absl::Status HloEvaluator::HandleMap(const HloInstruction* map) {
  auto operands = map->operands();
  const HloComputation* computation = map->to_apply();

  Literal result(map->shape());

  HloEvaluator embedded_evaluator(max_loop_iterations_);
  TF_RETURN_IF_ERROR(
      Apply<PopulateImpl>(result, [&](absl::Span<const int64_t> multi_index) {
        std::vector<Literal> arg_literals;
        arg_literals.reserve(operands.size());

        // Construct scalar literal parameters to be passed to the map
        // computation.
        for (auto operand : operands) {
          const Literal& arg_literal = GetEvaluatedLiteralFor(operand);
          arg_literals.push_back(
              LiteralUtil::GetScalarLiteral(arg_literal, multi_index));
        }

        Literal computed_result =
            embedded_evaluator.Evaluate(*computation, arg_literals).value();
        // Clear visit states so that the we can use the evaluate again on
        // the same computation.
        embedded_evaluator.ResetVisitStates();

        return computed_result;
      }));
  evaluated_[map] = std::move(result);
  return absl::OkStatus();
}

absl::Status HloEvaluator::HandleCustomCall(const HloInstruction* custom_call) {
  if (!custom_call_handler_) {
    // No handler is registered; this means custom-calls are not allowed.
    return DefaultAction(custom_call);
  }

  // Evaluate input operands so the handler has access to the operand data.
  std::vector<const Literal*> operands;
  operands.reserve(custom_call->operand_count());
  for (const HloInstruction* operand : custom_call->operands()) {
    operands.push_back(&GetEvaluatedLiteralFor(operand));
  }

  // Synchronously issue the handler to populate the instruction output literal.
  TF_ASSIGN_OR_RETURN(
      auto output, custom_call_handler_(custom_call, absl::MakeSpan(operands)));

  evaluated_[custom_call] = std::move(output);
  return absl::OkStatus();
}

absl::Status HloEvaluator::Preprocess(const HloInstruction* hlo) {
  VLOG(3) << "About to visit HLO: " << hlo->ToString();
  if (!enable_partial_evaluation_) {
    for (const HloInstruction* operand : hlo->operands()) {
      if (!IsAlreadyEvaluated(operand) ||
          !GetEvaluatedLiteralFor(operand).IsKnown()) {
        return absl::FailedPreconditionError(
            "Failed to evaluate instruction since its operands are unknown "
            "or undetermined and partial evaluation is not enabled.");
      }
    }
  }
  return ShapeUtil::ValidateShapeWithOptionalLayout(hlo->shape());
}

absl::Status HloEvaluator::Postprocess(const HloInstruction* hlo) {
  VLOG(3) << "Finished visiting " << hlo->ToString()
          << "; evaluated value is: " << GetEvaluatedLiteralFor(hlo).ToString();
  // Out of convenience the literal may have been produced with a different
  // layout. Relayout as indicated by the HLO instruction.
  auto evaluated_shape = GetEvaluatedLiteralFor(hlo).shape();
  Shape hlo_shape = hlo->shape();
  if (hlo_shape.IsArray() && !hlo_shape.has_layout()) {
    *hlo_shape.mutable_layout() =
        LayoutUtil::GetDefaultLayoutForShape(hlo_shape);
  }
  if (evaluated_shape.has_layout() && hlo_shape.has_layout() &&
      !Layout::Equal().MinorToMajorOnly()(evaluated_shape.layout(),
                                          hlo_shape.layout())) {
    evaluated_.at(hlo) = evaluated_.at(hlo).Relayout(hlo_shape);
  }
  return absl::OkStatus();
}

}  // namespace zkx
