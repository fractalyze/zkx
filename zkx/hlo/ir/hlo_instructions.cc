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

#include "zkx/hlo/ir/hlo_instructions.h"

#include "absl/algorithm/container.h"

#include "zkx/hlo/ir/hlo_casting_utils.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/protobuf_util.h"

namespace zkx {
namespace {

void SetThreadName(HloComputation* called_computation,
                   std::string_view execution_thread,
                   bool skip_async_execution_thread_overwrite) {
  called_computation->SetExecutionThread(execution_thread);
  for (HloInstruction* instr : called_computation->instructions()) {
    if (instr->IsAsynchronous()) {
      if (!skip_async_execution_thread_overwrite) {
        // Set async instruction thread name and also recursively set async
        // computations.
        instr->set_async_execution_thread(execution_thread);
      }
      continue;
    }
    for (HloComputation* nested_called_computation :
         instr->called_computations()) {
      SetThreadName(nested_called_computation, execution_thread,
                    skip_async_execution_thread_overwrite);
    }
  }
}

}  // namespace

bool HloDimensionsInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloDimensionsInstruction&>(other);
  return dimensions() == casted_other.dimensions();
}

HloInstructionProto HloDimensionsInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64_t dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

HloBroadcastInstruction::HloBroadcastInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> broadcast_dimension)
    : HloDimensionsInstruction(HloOpcode::kBroadcast, shape,
                               broadcast_dimension) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloBroadcastInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloBroadcastInstruction>(shape, new_operands[0],
                                                   dimensions());
}

HloFftInstruction::HloFftInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    FftType fft_type, int64_t fft_length, bool fft_do_bit_reverse)
    : HloInstruction(HloOpcode::kFft, shape),
      fft_type_(fft_type),
      fft_length_(fft_length),
      fft_do_bit_reverse_(fft_do_bit_reverse) {
  CHECK_LE(new_operands.size(), 2);
  AppendOperands(new_operands);
}

HloInstructionProto HloFftInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_fft_type(fft_type_);
  proto.set_fft_length(fft_length_);
  proto.set_fft_do_bit_reverse(fft_do_bit_reverse_);
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloFftInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   printer.Next([this](Printer* printer) {
//     AppendCat(printer, "fft_type=", FftType_Name(fft_type()));
//   });
//   printer.Next([this](Printer* printer) {
//     printer->Append("fft_length={");
//     AppendJoin(printer, fft_length(), ",");
//     printer->Append("}");
//   });
// }

bool HloFftInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloFftInstruction&>(other);
  return fft_type() == casted_other.fft_type() &&
         fft_length() == casted_other.fft_length() &&
         fft_do_bit_reverse() == casted_other.fft_do_bit_reverse();
}

std::unique_ptr<HloInstruction> HloFftInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloFftInstruction>(shape, new_operands, fft_type_,
                                             fft_length_, fft_do_bit_reverse_);
}

HloMsmInstruction::HloMsmInstruction(const Shape& shape,
                                     HloInstruction* scalars,
                                     HloInstruction* bases, int32_t window_bits)
    : HloInstruction(HloOpcode::kMsm, shape), window_bits_(window_bits) {
  AppendOperand(scalars);
  AppendOperand(bases);
}

HloInstructionProto HloMsmInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_window_bits(window_bits_);
  return proto;
}

bool HloMsmInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloMsmInstruction&>(other);
  return window_bits() == casted_other.window_bits();
}

std::unique_ptr<HloInstruction> HloMsmInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloMsmInstruction>(shape, new_operands[0],
                                             new_operands[1], window_bits_);
}

HloAsyncInstruction::HloAsyncInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands, HloOpcode async_wrapped_opcode)
    : HloInstruction(opcode, shape) {
  CHECK(opcode == HloOpcode::kAsyncStart || operands.size() == 1);
  for (auto operand : operands) {
    AppendOperand(operand);
  }

  // Drop 'async' from async-{start/update/done} to get the suffix.
  std::string_view suffix = HloOpcodeString(opcode).substr(5);
  std::string_view wrapped_name = HloOpcodeString(async_wrapped_opcode);
  SetAndSanitizeName(absl::StrCat(wrapped_name, suffix));
}

HloAsyncInstruction::HloAsyncInstruction(HloOpcode opcode, const Shape& shape,
                                         HloInstruction* operand)
    : HloAsyncInstruction(opcode, shape, absl::MakeConstSpan(&operand, 1),
                          operand->async_wrapped_opcode()) {
  CHECK(operand->opcode() == HloOpcode::kAsyncStart ||
        operand->opcode() == HloOpcode::kAsyncUpdate);
  HloAsyncInstruction* prev = Cast<HloAsyncInstruction>(operand);
  prev->async_chain_next_ = this;
}

HloComputation* HloAsyncInstruction::async_wrapped_computation() const {
  return async_chain_start()->called_computations().front();
}

HloInstruction* HloAsyncInstruction::async_wrapped_instruction() const {
  return async_chain_start()->async_wrapped_computation()->root_instruction();
}

HloOpcode HloAsyncInstruction::async_wrapped_opcode() const {
  return async_chain_start()->async_wrapped_instruction()->opcode();
}

std::string_view HloAsyncInstruction::async_execution_thread() const {
  return async_chain_start()->async_execution_thread();
}

HloAsyncInstruction* HloAsyncInstruction::async_chain_start() const {
  if (opcode() == HloOpcode::kAsyncStart) {
    return const_cast<HloAsyncInstruction*>(this);
  }

  HloInstruction* prev = operands()[0];
  while (prev->opcode() != HloOpcode::kAsyncStart) {
    // If the prev op in the chain isn't async-start, it must be async-update.
    CHECK(prev->opcode() == HloOpcode::kAsyncUpdate);
    prev = prev->operands()[0];
  }
  return Cast<HloAsyncInstruction>(prev);
}

HloAsyncInstruction* HloAsyncInstruction::async_chain_done() const {
  if (opcode() == HloOpcode::kAsyncDone) {
    return const_cast<HloAsyncInstruction*>(this);
  }

  HloAsyncInstruction* next = async_chain_next_;
  while (next->opcode() != HloOpcode::kAsyncDone) {
    // If the next op in the chain isn't async-done, it must be async-update.
    CHECK(next->opcode() == HloOpcode::kAsyncUpdate);
    next = next->async_chain_next_;
  }
  return next;
}

std::vector<HloAsyncInstruction*> HloAsyncInstruction::GetAsyncChain() const {
  std::vector<HloAsyncInstruction*> chain;
  HloAsyncInstruction* current = async_chain_start();
  do {
    chain.push_back(current);
    current = current->async_chain_next_;
  } while (current != nullptr);
  return chain;
}

bool HloAsyncInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  return opcode() == other.opcode() &&
         eq_computations(async_wrapped_computation(),
                         other.async_wrapped_computation());
}

std::unique_ptr<HloInstruction> HloAsyncInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloAsyncInstruction>(opcode(), shape,
                                               new_operands[0]);
}

HloAsyncStartInstruction::HloAsyncStartInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    HloComputation* async_computation, std::string_view async_execution_thread)
    : HloAsyncInstruction(opcode, shape, operands,
                          async_computation->root_instruction()->opcode()) {
  CHECK(!async_computation->IsCustomCallComputation());
  CHECK(!async_computation->IsFusionComputation());
  CHECK(!async_computation->IsAsyncComputation());
  AppendComputation(async_computation);
  async_computation->AddAsyncStart(this);
  HloAsyncStartInstruction::set_async_execution_thread(async_execution_thread);
}

HloAsyncStartInstruction::~HloAsyncStartInstruction() {
  ClearAsyncComputationInstruction();
}

void HloAsyncStartInstruction::ClearCalledComputations() {
  ClearAsyncComputationInstruction();
  HloInstruction::ClearCalledComputations();
}

void HloAsyncStartInstruction::ClearAsyncComputationInstruction() {
  // Each async instruction calls a single computation, but we use
  // called_computations() instead of async_wrapped_instruction(), because the
  // order in which things get destructed can vary; the async computation's
  // back-pointer may already be null, which violates a check in
  // async_wrapped_instruction.
  if (!called_computations().empty() &&
      async_wrapped_computation()->AsyncStart() == this) {
    async_wrapped_computation()->RemoveAsyncStart();
  }
}

void HloAsyncStartInstruction::set_async_execution_thread(
    std::string_view async_execution_thread) {
  async_execution_thread_ = std::string(async_execution_thread);
  SetThreadName(async_wrapped_computation(), async_execution_thread,
                /*skip_async_execution_thread_overwrite=*/false);
}

HloInstructionProto HloAsyncStartInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_async_execution_thread(async_execution_thread_ ==
                                           HloInstruction::kMainExecutionThread
                                       ? ""
                                       : async_execution_thread_);
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloAsyncStartInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   if (async_execution_thread_ != kMainExecutionThread) {
//     printer.Next([this](Printer* printer) {
//       AppendCat(printer, "async_execution_thread=\"",
//       async_execution_thread_,
//                 "\"");
//     });
//   }
//   if (options.syntax_sugar_async_ops() &&
//       async_wrapped_computation()->CanExpandIntoSingleInstruction()) {
//     async_wrapped_instruction()->PrintExtraAttributes(printer, options);
//   }
// }

std::unique_ptr<HloInstruction>
HloAsyncStartInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  HloComputation* new_wrapped_computation = nullptr;
  if (context != nullptr) {
    new_wrapped_computation =
        context->FindComputation(async_wrapped_computation());
  }
  if (new_wrapped_computation == nullptr) {
    HloModule* module = context != nullptr ? context->module() : GetModule();
    new_wrapped_computation = module->AddEmbeddedComputation(
        async_wrapped_computation()->Clone("clone", context));
  }

  return std::make_unique<HloAsyncStartInstruction>(
      opcode(), shape, new_operands, new_wrapped_computation,
      async_execution_thread_);
}

HloCopyStartInstruction::HloCopyStartInstruction(
    const Shape& shape, HloInstruction* operand,
    std::optional<int> cross_program_prefetch_index)
    : HloInstruction(HloOpcode::kCopyStart, shape),
      cross_program_prefetch_index_(cross_program_prefetch_index) {
  AppendOperand(operand);
}

HloInstructionProto HloCopyStartInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (cross_program_prefetch_index_.has_value()) {
    proto.set_cross_program_prefetch_index(*cross_program_prefetch_index_);
  }
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloCopyStartInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   if (cross_program_prefetch_index_.has_value()) {
//     printer.Next([this](Printer* printer) {
//       AppendCat(printer, "cross_program_prefetch_index=",
//                 *cross_program_prefetch_index_);
//     });
//   }
// }

bool HloCopyStartInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloCopyStartInstruction&>(other);
  return cross_program_prefetch_index() ==
         casted_other.cross_program_prefetch_index();
}

std::unique_ptr<HloInstruction>
HloCopyStartInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloCopyStartInstruction>(
      shape, new_operands[0], cross_program_prefetch_index());
}

HloCompareInstruction::HloCompareInstruction(const Shape& shape,
                                             HloInstruction* lhs,
                                             HloInstruction* rhs,
                                             ComparisonDirection direction,
                                             std::optional<PrimitiveType> type)
    : HloInstruction(HloOpcode::kCompare, shape),
      compare_(type.has_value()
                   ? Comparison(direction, *type)
                   : Comparison(direction, lhs->shape().element_type())) {
  AppendOperand(lhs);
  AppendOperand(rhs);
}

HloInstructionProto HloCompareInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_comparison_direction(
      ComparisonDirectionToString(compare_.GetDirection()));
  proto.set_comparison_primitive_type(compare_.GetPrimitiveType());
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloCompareInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   printer.Next([this](Printer* printer) {
//     AppendCat(printer, "direction=",
//     ComparisonDirectionToString(direction()));
//   });
//   if (compare_.GetType() !=
//       Comparison::DefaultComparisonType(operand(0)->shape().element_type()))
//       {
//     printer.Next([this](Printer* printer) {
//       AppendCat(printer, "type=",
//       ComparisonTypeToString(compare_.GetType()));
//     });
//   }
// }

bool HloCompareInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloCompareInstruction&>(other);
  return direction() == casted_other.direction();
}

std::unique_ptr<HloInstruction> HloCompareInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloCompareInstruction>(
      shape, new_operands[0], new_operands[1], direction(), primitive_type());
}

HloChannelInstruction::HloChannelInstruction(
    HloOpcode opcode, const Shape& shape,
    const std::optional<int64_t>& channel_id)
    : HloInstruction(opcode, shape), channel_id_(channel_id) {}

void HloChannelInstruction::set_channel_id(
    const std::optional<int64_t>& channel_id) {
  channel_id_ = channel_id;
}

HloInstructionProto HloChannelInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (channel_id_) {
    proto.set_channel_id(*channel_id_);
  }
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloChannelInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& /*options*/) const {
//   if (!channel_id_) return;
//   printer.Next([this](Printer* printer) {
//     AppendCat(printer, "channel_id=", *channel_id_);
//   });
// }

bool HloChannelInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  if (!IdenticalSlowPathIgnoringChannelIdValues(other, eq_computations)) {
    return false;
  }
  const auto& casted_other = static_cast<const HloChannelInstruction&>(other);
  return channel_id() == casted_other.channel_id();
}

HloSendRecvInstruction::HloSendRecvInstruction(
    HloOpcode opcode, const Shape& shape, std::optional<int64_t> channel_id,
    bool is_host_transfer)
    : HloChannelInstruction(opcode, shape, channel_id),
      is_host_transfer_(is_host_transfer) {}

HloInstructionProto HloSendRecvInstruction::ToProto() const {
  HloInstructionProto proto = HloChannelInstruction::ToProto();
  proto.set_is_host_transfer(is_host_transfer_);
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloSendRecvInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   HloChannelInstruction::PrintExtraAttributesImpl(printer, options);
//   if (is_host_transfer()) {
//     printer.Next(
//         [](Printer* printer) { printer->Append("is_host_transfer=true"); });
//   }
// }

bool HloSendRecvInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  // Not yet supported.
  return false;
}

// Send instruction produces a tuple of {aliased operand, U32 context}.
HloSendInstruction::HloSendInstruction(HloInstruction* operand,
                                       HloInstruction* token,
                                       std::optional<int64_t> channel_id,
                                       bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kSend,
          ShapeUtil::MakeTupleShape({CHECK_NOTNULL(operand)->shape(),
                                     ShapeUtil::MakeShape(U32, {}),
                                     ShapeUtil::MakeTokenShape()}),
          channel_id, is_host_transfer) {
  AppendOperand(operand);
  AppendOperand(token);
}

std::unique_ptr<HloInstruction> HloSendInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloSendInstruction>(new_operands[0], new_operands[1],
                                              channel_id(), is_host_transfer());
}

HloSendDoneInstruction::HloSendDoneInstruction(HloSendInstruction* operand,
                                               bool is_host_transfer)
    : HloSendRecvInstruction(HloOpcode::kSendDone, ShapeUtil::MakeTokenShape(),
                             operand->channel_id(), is_host_transfer) {
  AppendOperand(operand);
}

HloSendDoneInstruction::HloSendDoneInstruction(
    HloInstruction* operand, std::optional<int64_t> channel_id,
    bool is_host_transfer)
    : HloSendRecvInstruction(HloOpcode::kSendDone, ShapeUtil::MakeTokenShape(),
                             channel_id, is_host_transfer) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloSendDoneInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  HloSendInstruction* send = dynamic_cast<HloSendInstruction*>(new_operands[0]);
  if (send != nullptr) {
    return std::make_unique<HloSendDoneInstruction>(send, is_host_transfer());
  }

  return std::make_unique<HloSendDoneInstruction>(new_operands[0], channel_id(),
                                                  is_host_transfer());
}

// Recv instruction produces a tuple of {receive buffer, U32 context}.
HloRecvInstruction::HloRecvInstruction(const Shape& shape,
                                       HloInstruction* token,
                                       std::optional<int64_t> channel_id,
                                       bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kRecv,
          ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U32, {}),
                                     ShapeUtil::MakeTokenShape()}),
          channel_id, is_host_transfer) {
  AppendOperand(token);
}

std::unique_ptr<HloInstruction> HloRecvInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloRecvInstruction>(
      ShapeUtil::GetTupleElementShape(shape, 0), new_operands[0], channel_id(),
      is_host_transfer());
}

HloRecvDoneInstruction::HloRecvDoneInstruction(HloRecvInstruction* operand,
                                               bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kRecvDone,
          ShapeUtil::MakeTupleShape(
              {ShapeUtil::GetTupleElementShape(operand->shape(), 0),
               ShapeUtil::MakeTokenShape()}),
          operand->channel_id(), is_host_transfer) {
  AppendOperand(operand);
}

HloRecvDoneInstruction::HloRecvDoneInstruction(
    HloInstruction* operand, std::optional<int64_t> channel_id,
    bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kRecvDone,
          ShapeUtil::MakeTupleShape(
              {ShapeUtil::GetTupleElementShape(operand->shape(), 0),
               ShapeUtil::MakeTokenShape()}),
          channel_id, is_host_transfer) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloRecvDoneInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  HloRecvInstruction* recv = dynamic_cast<HloRecvInstruction*>(new_operands[0]);
  if (recv != nullptr) {
    return std::make_unique<HloRecvDoneInstruction>(recv, is_host_transfer());
  }

  return std::make_unique<HloRecvDoneInstruction>(new_operands[0], channel_id(),
                                                  is_host_transfer());
}

HloCollectiveInstruction::HloCollectiveInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    const CollectiveDeviceList& device_list, bool constrain_layout,
    const std::optional<int64_t>& channel_id)
    : HloChannelInstruction(opcode, shape, channel_id),
      device_list_(device_list),
      constrain_layout_(constrain_layout) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

HloInstructionProto HloCollectiveInstruction::ToProto() const {
  HloInstructionProto proto = HloChannelInstruction::ToProto();
  *proto.mutable_collective_device_list() = device_list_.ToProto();
  proto.set_constrain_layout(constrain_layout_);
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloCollectiveInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   HloChannelInstruction::PrintExtraAttributesImpl(printer, options);
//   printer.Next([this, &options](Printer* printer) {
//     VLOG(4) << name() << " replica_groups="
//             <<
//             device_list_.ToString(options.print_full_replica_group_list());

//     AppendCat(printer, "replica_groups=",
//               device_list_.ToString(options.print_full_replica_group_list()));
//   });
//   if (constrain_layout_) {
//     printer.Next(
//         [](Printer* printer) { printer->Append("constrain_layout=true"); });
//   }
// }

bool HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloCollectiveInstruction&>(other);
  return HloChannelInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         constrain_layout() == casted_other.constrain_layout() &&
         absl::c_equal(replica_groups(), casted_other.replica_groups(),
                       [](const ReplicaGroup& a, const ReplicaGroup& b) {
                         return absl::c_equal(a.replica_ids(), b.replica_ids());
                       });
}

HloAllGatherInstruction::HloAllGatherInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands, int64_t all_gather_dimension,
    const CollectiveDeviceList& device_list, bool constrain_layout,
    const std::optional<int64_t>& channel_id, bool use_global_device_ids)
    : HloCollectiveInstruction(opcode, shape, operands, device_list,
                               constrain_layout, channel_id),
      all_gather_dimension_(all_gather_dimension),
      use_global_device_ids_(use_global_device_ids) {}

HloAllGatherInstruction::HloAllGatherInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands, int64_t all_gather_dimension,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const std::optional<int64_t>& channel_id, bool use_global_device_ids)
    : HloAllGatherInstruction(opcode, shape, operands, all_gather_dimension,
                              CollectiveDeviceList(replica_groups),
                              constrain_layout, channel_id,
                              use_global_device_ids) {}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloAllGatherInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   HloCollectiveInstruction::PrintExtraAttributesImpl(printer, options);
//   printer.Next([this](Printer* printer) {
//     AppendCat(printer, "dimensions={", all_gather_dimension_, "}");
//   });
//   if (use_global_device_ids_) {
//     printer.Next([](Printer* printer) {
//       printer->Append("use_global_device_ids=true");
//     });
//   }
// }

std::unique_ptr<HloInstruction>
HloAllGatherInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloAllGatherInstruction>(
      opcode(), shape, new_operands, all_gather_dimension(), device_list(),
      constrain_layout(), channel_id(), use_global_device_ids());
}

HloInstructionProto HloAllGatherInstruction::ToProto() const {
  HloInstructionProto proto = HloCollectiveInstruction::ToProto();
  proto.add_dimensions(all_gather_dimension_);
  proto.set_use_global_device_ids(use_global_device_ids_);
  return proto;
}

bool HloAllGatherInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloAllGatherInstruction&>(other);
  return HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         all_gather_dimension_ == casted_other.all_gather_dimension() &&
         use_global_device_ids() == casted_other.use_global_device_ids();
}

HloAllReduceInstructionBase::HloAllReduceInstructionBase(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation, const CollectiveDeviceList& device_list,
    bool constrain_layout, const std::optional<int64_t>& channel_id,
    bool use_global_device_ids)
    : HloCollectiveInstruction(opcode, shape, operands, device_list,
                               constrain_layout, channel_id),
      use_global_device_ids_(use_global_device_ids) {
  AppendComputation(reduce_computation);
  reduce_computation->SetCollectiveCallInstruction(this);
}

HloInstructionProto HloAllReduceInstructionBase::ToProto() const {
  HloInstructionProto proto = HloCollectiveInstruction::ToProto();
  proto.set_use_global_device_ids(use_global_device_ids_);
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloAllReduceInstructionBase::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   HloCollectiveInstruction::PrintExtraAttributesImpl(printer, options);
//   if (use_global_device_ids_) {
//     printer.Next([](Printer* printer) {
//       printer->Append("use_global_device_ids=true");
//     });
//   }
// }

bool HloAllReduceInstructionBase::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  if (opcode() != other.opcode()) {
    return false;
  }
  const auto& casted_other =
      static_cast<const HloAllReduceInstructionBase&>(other);
  return HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         constrain_layout() == casted_other.constrain_layout() &&
         use_global_device_ids() == casted_other.use_global_device_ids() &&
         eq_computations(to_apply(), casted_other.to_apply());
}

bool HloAllReduceInstruction::IsNoop() const {
  for (const auto& replica_group : replica_groups()) {
    if (replica_group.replica_ids().size() != 1) {
      return false;
    }
  }
  return !channel_id();
}

std::unique_ptr<HloInstruction>
HloAllReduceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloAllReduceInstruction>(
      opcode(), shape, new_operands, to_apply(), device_list(),
      constrain_layout(), channel_id(), use_global_device_ids());
}

HloReduceScatterInstruction::HloReduceScatterInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation, const CollectiveDeviceList& device_list,
    bool constrain_layout, const std::optional<int64_t>& channel_id,
    bool use_global_device_ids, int64_t scatter_dimension)
    : HloAllReduceInstructionBase(
          HloOpcode::kReduceScatter, shape, operands, reduce_computation,
          device_list, constrain_layout, channel_id, use_global_device_ids),
      scatter_dimension_(scatter_dimension) {}

HloReduceScatterInstruction::HloReduceScatterInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const std::optional<int64_t>& channel_id, bool use_global_device_ids,
    int64_t scatter_dimension)
    : HloReduceScatterInstruction(shape, operands, reduce_computation,
                                  CollectiveDeviceList(replica_groups),
                                  constrain_layout, channel_id,
                                  use_global_device_ids, scatter_dimension) {}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloReduceScatterInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   HloAllReduceInstructionBase::PrintExtraAttributesImpl(printer, options);
//   printer.Next([this](Printer* printer) {
//     AppendCat(printer, "dimensions={", scatter_dimension_, "}");
//   });
// }

HloInstructionProto HloReduceScatterInstruction::ToProto() const {
  HloInstructionProto proto = HloAllReduceInstructionBase::ToProto();
  proto.add_dimensions(scatter_dimension_);
  return proto;
}

bool HloReduceScatterInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloReduceScatterInstruction&>(other);
  return HloAllReduceInstructionBase::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         scatter_dimension_ == casted_other.scatter_dimension();
}

std::unique_ptr<HloInstruction>
HloReduceScatterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloReduceScatterInstruction>(
      shape, new_operands, to_apply(), device_list(), constrain_layout(),
      channel_id(), use_global_device_ids(), scatter_dimension());
}

HloAllToAllInstruction::HloAllToAllInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const CollectiveDeviceList& device_list, bool constrain_layout,
    const std::optional<int64_t>& channel_id,
    const std::optional<int64_t>& split_dimension)
    : HloCollectiveInstruction(HloOpcode::kAllToAll, shape, operands,
                               device_list, constrain_layout, channel_id),
      split_dimension_(split_dimension) {}

HloAllToAllInstruction::HloAllToAllInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const std::optional<int64_t>& channel_id,
    const std::optional<int64_t>& split_dimension)
    : HloAllToAllInstruction(shape, operands,
                             CollectiveDeviceList(replica_groups),
                             constrain_layout, channel_id, split_dimension) {}

std::unique_ptr<HloInstruction>
HloAllToAllInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloAllToAllInstruction>(
      shape, new_operands, device_list(), constrain_layout(), channel_id(),
      split_dimension());
}

HloInstructionProto HloAllToAllInstruction::ToProto() const {
  HloInstructionProto proto = HloCollectiveInstruction::ToProto();
  if (split_dimension_) {
    proto.add_dimensions(*split_dimension_);
  }
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloAllToAllInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   HloCollectiveInstruction::PrintExtraAttributesImpl(printer, options);
//   if (split_dimension_) {
//     printer.Next([this](Printer* printer) {
//       AppendCat(printer, "dimensions={", *split_dimension_, "}");
//     });
//   }
// }

bool HloAllToAllInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloAllToAllInstruction&>(other);
  return HloCollectiveInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         split_dimension_ == casted_other.split_dimension();
}

HloRaggedAllToAllInstruction::HloRaggedAllToAllInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const CollectiveDeviceList& device_list,
    const std::optional<int64_t>& channel_id)
    : HloCollectiveInstruction(HloOpcode::kRaggedAllToAll, shape, operands,
                               device_list,
                               /*constrain_layout=*/false, channel_id) {}

HloRaggedAllToAllInstruction::HloRaggedAllToAllInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    absl::Span<const ReplicaGroup> replica_groups,
    const std::optional<int64_t>& channel_id)
    : HloRaggedAllToAllInstruction(
          shape, operands, CollectiveDeviceList(replica_groups), channel_id) {}

std::unique_ptr<HloInstruction>
HloRaggedAllToAllInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloRaggedAllToAllInstruction>(
      shape, new_operands, device_list(), channel_id());
}

HloInstructionProto HloRaggedAllToAllInstruction::ToProto() const {
  return HloCollectiveInstruction::ToProto();
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloRaggedAllToAllInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   HloCollectiveInstruction::PrintExtraAttributesImpl(printer, options);
// }

HloCollectiveBroadcastInstruction::HloCollectiveBroadcastInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    const CollectiveDeviceList& device_list, bool constrain_layout,
    const std::optional<int64_t>& channel_id)
    : HloCollectiveInstruction(opcode, shape, operands, device_list,
                               constrain_layout, channel_id) {}

HloCollectiveBroadcastInstruction::HloCollectiveBroadcastInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
    const std::optional<int64_t>& channel_id)
    : HloCollectiveBroadcastInstruction(opcode, shape, operands,
                                        CollectiveDeviceList(replica_groups),
                                        constrain_layout, channel_id) {}

HloInstructionProto HloCollectiveBroadcastInstruction::ToProto() const {
  return HloCollectiveInstruction::ToProto();
}

std::unique_ptr<HloInstruction>
HloCollectiveBroadcastInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return std::make_unique<HloCollectiveBroadcastInstruction>(
      opcode(), shape, new_operands, device_list(), constrain_layout(),
      channel_id());
}

HloCollectivePermuteInstruction::HloCollectivePermuteInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    const std::optional<int64_t>& channel_id)
    : HloChannelInstruction(opcode, shape, channel_id),
      source_target_pairs_(source_target_pairs) {
  AppendOperands(operands);
  inplace_ = false;
}

HloCollectivePermuteInstruction::HloCollectivePermuteInstruction(
    HloOpcode opcode, const Shape& shape, HloInstruction* input,
    HloInstruction* output, HloInstruction* input_start_indices,
    HloInstruction* output_start_indices,
    absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
    absl::Span<const std::vector<int64_t>> slice_sizes,
    const std::optional<int64_t>& channel_id)
    : HloChannelInstruction(opcode, shape, channel_id),
      source_target_pairs_(source_target_pairs.begin(),
                           source_target_pairs.end()),
      slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
  AppendOperand(input);
  AppendOperand(output);
  AppendOperand(input_start_indices);
  AppendOperand(output_start_indices);
  inplace_ = true;
}

HloInstructionProto HloCollectivePermuteInstruction::ToProto() const {
  HloInstructionProto proto = HloChannelInstruction::ToProto();
  for (const auto& pair : source_target_pairs()) {
    auto* proto_pair = proto.add_source_target_pairs();
    proto_pair->set_source(pair.first);
    proto_pair->set_target(pair.second);
  }
  for (const auto& slice_size : dynamic_slice_sizes_list()) {
    for (const auto& dimension_slice_size : slice_size) {
      proto.add_dynamic_slice_sizes(dimension_slice_size);
    }
  }
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloCollectivePermuteInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   HloChannelInstruction::PrintExtraAttributesImpl(printer, options);
//   printer.Next([this](Printer* printer) {
//     printer->Append("source_target_pairs={");
//     AppendJoin(printer, source_target_pairs(), ",",
//                [](Printer* printer, const std::pair<int64_t, int64_t>& pair)
//                {
//                  AppendCat(printer, "{", pair.first, ",", pair.second);
//                  printer->Append("}");
//                });
//     printer->Append("}");
//   });
//   if (!dynamic_slice_sizes_list().empty()) {
//     printer.Next([this](Printer* printer) {
//       printer->Append("slice_sizes={");
//       AppendJoin(printer, dynamic_slice_sizes_list(), ",",
//                  [](Printer* printer, const std::vector<int64_t>&
//                  slice_sizes) {
//                    printer->Append("{");
//                    AppendJoin(printer, slice_sizes, ",");
//                    printer->Append("}");
//                  });
//       printer->Append("}");
//     });
//   }
// }

bool HloCollectivePermuteInstruction::IdenticalSlowPathIgnoringChannelIdValues(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  if (opcode() != other.opcode()) {
    return false;
  }
  const auto& casted_other =
      static_cast<const HloCollectivePermuteInstruction&>(other);
  return HloChannelInstruction::IdenticalSlowPathIgnoringChannelIdValues(
             other, eq_computations) &&
         absl::c_equal(
             source_target_pairs(), casted_other.source_target_pairs(),
             [](const std::pair<int64_t, int64_t>& a,
                const std::pair<int64_t, int64_t>& b) { return a == b; }) &&
         absl::c_equal(
             dynamic_slice_sizes_list(),
             casted_other.dynamic_slice_sizes_list(),
             [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
               return absl::c_equal(a, b);
             });
}

std::unique_ptr<HloInstruction>
HloCollectivePermuteInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  if (dynamic_slice_sizes_list().empty()) {
    return std::make_unique<HloCollectivePermuteInstruction>(
        opcode(), shape,
        absl::Span<HloInstruction* const>(new_operands.subspan(0, 1)),
        source_target_pairs(), channel_id());
  } else {
    return std::make_unique<HloCollectivePermuteInstruction>(
        opcode(), shape, new_operands[0], new_operands[1], new_operands[2],
        new_operands[3], source_target_pairs(), dynamic_slice_sizes_list(),
        channel_id());
  }
}

HloReverseInstruction::HloReverseInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> dimensions)
    : HloDimensionsInstruction(HloOpcode::kReverse, shape, dimensions) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction> HloReverseInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloReverseInstruction>(shape, new_operands[0],
                                                 dimensions());
}

HloConcatenateInstruction::HloConcatenateInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    int64_t dimension)
    : HloDimensionsInstruction(HloOpcode::kConcatenate, shape, {dimension}) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

std::unique_ptr<HloInstruction>
HloConcatenateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return std::make_unique<HloConcatenateInstruction>(shape, new_operands,
                                                     concatenate_dimension());
}

HloSliceInstruction::HloSliceInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64_t> start_indices,
    absl::Span<const int64_t> limit_indices, absl::Span<const int64_t> strides)
    : HloInstruction(HloOpcode::kSlice, shape),
      slice_starts_(start_indices.begin(), start_indices.end()),
      slice_limits_(limit_indices.begin(), limit_indices.end()),
      slice_strides_(strides.begin(), strides.end()) {
  AppendOperand(operand);
  // For backward compatibility with old serialized computations: if there are
  // no strides, assume all strides are 1.
  // TODO(b/63317920): remove this code.
  if (slice_strides_.empty()) {
    slice_strides_ = std::vector<int64_t>(start_indices.size(), 1LL);
  }
}

HloInstructionProto HloSliceInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int i = 0; i < slice_starts_.size(); ++i) {
    auto* slice_dimension = proto.add_slice_dimensions();
    slice_dimension->set_start(slice_starts_[i]);
    slice_dimension->set_limit(slice_limits_[i]);
    slice_dimension->set_stride(slice_strides_[i]);
  }
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloSliceInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   printer.Next([this](Printer* printer) {
//     const bool omit_stride = absl::c_all_of(
//         slice_strides_, [](int64_t stride) { return stride == 1; });
//     printer->Append("slice={");
//     AppendJoin(printer, slice_starts_, ", ",
//                [&](Printer* printer, auto& slice_start) {
//                  const auto i = &slice_start - slice_starts_.data();
//                  AppendCat(printer, "[", slice_start, ":", slice_limits_[i]);
//                  if (!omit_stride) {
//                    AppendCat(printer, ":", slice_strides_[i]);
//                  }
//                  printer->Append("]");
//                });
//     printer->Append("}");
//   });
// }

bool HloSliceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& other_slice = static_cast<const HloSliceInstruction&>(other);
  return slice_starts_ == other_slice.slice_starts_ &&
         slice_limits_ == other_slice.slice_limits_ &&
         slice_strides_ == other_slice.slice_strides_;
}

std::unique_ptr<HloInstruction> HloSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloSliceInstruction>(
      shape, new_operands[0], slice_starts_, slice_limits_, slice_strides_);
}

HloConstantInstruction::HloConstantInstruction(Literal literal)
    : HloInstruction(HloOpcode::kConstant, literal.shape()),
      literal_(new Literal(std::move(literal))) {}

HloConstantInstruction::HloConstantInstruction(Literal literal,
                                               const Shape& shape)
    : HloInstruction(HloOpcode::kConstant, shape),
      literal_(new Literal(std::move(literal))) {}

HloConstantInstruction::HloConstantInstruction(std::shared_ptr<Literal> literal,
                                               const Shape& shape)
    : HloInstruction(HloOpcode::kConstant, shape), literal_(literal) {}

HloConstantInstruction::HloConstantInstruction(const Shape& shape)
    : HloInstruction(HloOpcode::kConstant, shape) {}

HloInstructionProto HloConstantInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (literal_) {
    *proto.mutable_literal() = literal_->ToProto();
  }
  return proto;
}

bool HloConstantInstruction::IsElementwiseImpl(
    const std::optional<int64_t>& operand_idx) const {
  return true;
}

bool HloConstantInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  // TODO(chokobole): Uncomment this. Dependency: HloSliceInstruction
  // const auto& other_slice = static_cast<const HloSliceInstruction&>(other);
  // return literal() == other_slice.literal();
  return false;
}

std::unique_ptr<HloInstruction>
HloConstantInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  if (!literal_) {
    return std::make_unique<HloConstantInstruction>(this->shape());
  }
  // Literal's shape may have no/different tiling info. Use this instruction's
  // shape instead.
  CHECK(Shape::Equal().MinorToMajorOnlyInLayout()(literal_->shape(),
                                                  this->shape()));
  return std::make_unique<HloConstantInstruction>(literal_, this->shape());
}

void HloConstantInstruction::PrintOperandsWithCanonicalNameMap(
    Printer* printer, const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  if (options.print_only_essential_constants()) {
    if (!literal_) {
      printer->Append("{...}");
      return;
    }
    if (literal().IsAll(0)) {
      printer->Append("0");
      return;
    }
    if (literal().IsAll(1)) {
      printer->Append("1");
      return;
    }
    if (shape().IsInteger()) {
      // The following prevents high compilation latencies caused by serializing
      // large constant tensors; for example: b/265669625. The limit of 500k was
      // chosen empirically to make sure that serialization of the `literal_` is
      // less than a second.
      if (auto num_constants =
              absl::c_accumulate(shape().dimensions(), 1, std::multiplies<>());
          num_constants <= 500'000) {
        // TODO(chokobole): Uncomment this. Dependency: PrintWithoutShapeOneline
        // literal_->PrintWithoutShapeOneline(printer);
        return;
      }
    }
    printer->Append("{...}");
    return;
  }

  // For constants, show the actual value in place of an empty operand list.
  if (literal_ &&
      ((shape().IsArray() && ShapeUtil::ElementsIn(shape()) <= 10) ||
       options.print_large_constants())) {
    // Literal::ToString emits multidimensional arrays over multiple
    // lines. Compact this into one line by stripping out white space.
    // TODO(chokobole): Uncomment this. Dependency: PrintWithoutShapeOneline
    // literal_->PrintWithoutShapeOneline(printer);
  } else {
    // Do not show large constants or tuples.
    printer->Append("{...}");
  }
}

HloCallableInstruction::HloCallableInstruction(HloOpcode opcode,
                                               const Shape& shape)
    : HloInstruction(opcode, shape) {}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(HloOpcodeString(opcode));
}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    HloComputation* called_computation, std::string_view prefix)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(absl::StrCat(prefix, HloOpcodeString(opcode)));
  AppendComputation(called_computation);
}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    absl::Span<HloComputation* const> called_computations)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(HloOpcodeString(opcode));
  for (auto called_computation : called_computations) {
    AppendComputation(called_computation);
  }
}

HloCallableInstruction::HloCallableInstruction(HloOpcode opcode,
                                               const Shape& shape,
                                               const std::string& name,
                                               const std::string& attributes,
                                               int64_t version)
    : HloInstruction(opcode, shape) {
  auto frontend_attributes =
      BuildFrontendAttributesForComposite(name, attributes, version);
  add_frontend_attributes(frontend_attributes);
  set_is_composite(true);
}

HloCallableInstruction::HloCallableInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands, HloComputation* decomposition,
    const std::string& name, const std::string& attributes, int64_t version)
    : HloInstruction(opcode, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName(HloOpcodeString(opcode));
  AppendComputation(decomposition);

  auto frontend_attributes =
      BuildFrontendAttributesForComposite(name, attributes, version);
  add_frontend_attributes(frontend_attributes);
  set_is_composite(true);
}

HloCallableInstruction::~HloCallableInstruction() { ClearCalledComputations(); }

HloComputation* HloCallableInstruction::called_computation() const {
  CHECK(!called_computations().empty());
  return called_computations().front();
}

HloInstruction* HloCallableInstruction::called_computation_root() const {
  return called_computation()->root_instruction();
}

HloInstruction* HloCallableInstruction::AddCallOperand(
    HloInstruction* new_operand) {
  CHECK_EQ(operand_count(),
           called_computation()->parameter_instructions().size());
  const int64_t param_no = operand_count();
  std::string param_name = absl::StrCat("param_", param_no);
  HloInstruction* called_computation_parameter =
      called_computation()->AddParameter(HloInstruction::CreateParameter(
          param_no, new_operand->shape(), param_name));
  AppendOperand(new_operand);
  return called_computation_parameter;
}

HloInstruction* HloCallableInstruction::AppendInstructionIntoCalledComputation(
    HloInstruction* instruction_to_append, bool add_output) {
  // When add_output is false, this callable instruction must be a user of
  // instruction_to_append.
  if (!add_output) {
    CHECK(IsUserOf(instruction_to_append));
  }
  return CloneAndAppendInstructionIntoCalledComputation(instruction_to_append,
                                                        add_output);
}

HloInstruction*
HloCallableInstruction::CloneAndAppendInstructionIntoCalledComputation(
    HloInstruction* instruction_to_append, bool add_output) {
  VLOG(3) << "CloneAndAppendInstructionIntoCalledComputation:\n"
          << instruction_to_append->ToString();
  HloInstruction* clone = nullptr;
  bool do_not_clone =
      instruction_to_append->opcode() == HloOpcode::kTuple &&
      absl::c_all_of(instruction_to_append->users(), [](HloInstruction* u) {
        return u->opcode() == HloOpcode::kGetTupleElement;
      });
  if (called_computations().empty()) {
    // New fusion instruction. It should not be a multi-output instruction.
    CHECK(!add_output);
    auto builder = HloComputation::Builder(default_called_computation_name());
    builder.AddInstruction(instruction_to_append->Clone(/*suffix=*/""));
    auto* new_computation =
        CHECK_NOTNULL(GetModule())->AddEmbeddedComputation(builder.Build());
    AppendComputation(new_computation);
    if (opcode() == HloOpcode::kFusion) {
      new_computation->SetFusionInstruction(this);
    }

    clone = called_computation_root();
  } else {
    // When add_output is false, instruction_to_append is necessarily an
    // operand of the callable instruction. After appending this will no
    // longer be the case. Remove the operand from the operand list and remove
    // its corresponding called computation parameter instruction.
    bool in_operand_list =
        absl::c_linear_search(operands(), instruction_to_append);
    CHECK(add_output || in_operand_list);
    if (do_not_clone) {
      // We assume all uses of a kTuple operation are GTE ops. In this case,
      // we don't need to clone 'instruction_to_append'.
      CHECK(!in_operand_list);
      clone = instruction_to_append;
    } else {
      clone = called_computation()->AddInstruction(
          instruction_to_append->Clone(/*suffix=*/""));
    }
    const auto& called_computation_parameters =
        called_computation()->parameter_instructions();
    for (int64_t operand_num = 0; operand_num < operand_count();
         ++operand_num) {
      if (instruction_to_append == operand(operand_num)) {
        // Replace the called computation parameter instruction's uses with
        // the clone.
        HloInstruction* called_computation_parameter =
            called_computation_parameters[operand_num];
        CHECK_OK(called_computation_parameter->ReplaceAllUsesWith(clone));

        // Remove the corresponding called computation parameter and operand
        // from their respective vectors.
        CHECK_OK(called_computation()->RemoveParameter(operand_num));
        RemoveOperandAt(operand_num);
        break;
      }
    }
    // We've cloned instruction_to_append into this callable instruction, so
    // this callable instruction is no longer a use of instruction_to_append.
    if (in_operand_list) {
      DetachFrom(instruction_to_append);
      // When the instruction_to_append does not have other users, we don't
      // need to generate a multi-output instruction.
      if (instruction_to_append->user_count() == 0) {
        add_output = false;
      }
    }
  }

  // Reread the parameters in the computation.
  const auto& called_computation_parameters =
      called_computation()->parameter_instructions();

  // Add each operand of the clone as an operand of the callable instruction.
  // A complication is that some clone operands may already be operands of the
  // callable instruction.
  for (int64_t operand_num = 0; operand_num < clone->operand_count();
       ++operand_num) {
    HloInstruction* operand = clone->mutable_operand(operand_num);

    // See if this operand is already an operand of the callable instruction.
    CHECK_EQ(operands().size(), called_computation_parameters.size());
    HloInstruction* called_computation_parameter = nullptr;
    for (int64_t i = 0; i < operands().size(); ++i) {
      if (this->operand(i) == operand) {
        called_computation_parameter = called_computation_parameters[i];
        break;
      }
    }

    if (called_computation_parameter == nullptr) {
      // Clone's operand was not already an operand of the callable
      // instruction. Add it as an operand and add a corresponding called
      // computation parameter instruction.

      // No need to create an original value for an added parameter as the
      // original value is saved in the corresponding argument.
      called_computation_parameter = AddCallOperand(operand);
    }
    CHECK_OK(
        clone->ReplaceOperandWith(operand_num, called_computation_parameter));
  }

  if (clone != instruction_to_append) {
    // Copy over the original value to the clone of a fused instruction.
    if (auto original_value = instruction_to_append->original_value()) {
      clone->set_original_value(original_value);
    }
    VLOG(2) << "New clone:\n" << clone->ToString();
  }

  if (add_output) {
    int64_t user_count = instruction_to_append->user_count();
    CHECK(user_count > 0 || instruction_to_append->IsRoot())
        << "Unable to append instruction: " << instruction_to_append->ToString()
        << ", which has " << user_count << " users.";
    HloInstruction* root = called_computation_root();
    // Check whether we have replaced an existing fusion root with 'clone'. If
    // yes, no need to add a duplicate root.
    if (root->opcode() == HloOpcode::kTuple) {
      for (int64_t i = 0; i < root->operand_count(); ++i) {
        if (root->operand(i) == clone) {
          HloInstruction* new_gte = AddInstruction(
              HloInstruction::CreateGetTupleElement(clone->shape(), this, i));
          CHECK_OK(instruction_to_append->ReplaceAllUsesWith(new_gte));
          return clone;
        }
      }
    }
    // If this is already a multioutput instruction, expand the root tuple
    // by 1.
    HloInstruction::InstructionVector tuple_elements;
    bool newly_created_tuple_instr = false;
    if (root->opcode() == HloOpcode::kTuple) {
      tuple_elements = root->operands();
    } else {
      tuple_elements.push_back(root);
      newly_created_tuple_instr = true;
    }
    if (clone->opcode() == HloOpcode::kTuple) {
      for (auto inst : clone->operands()) {
        tuple_elements.push_back(inst);
      }
    } else {
      tuple_elements.push_back(clone);
    }
    HloInstruction* new_root = called_computation()->AddInstruction(
        HloInstruction::CreateTuple(tuple_elements));

    // No need to create an original value for a new root with added outputs
    // as the original value is saved in the get-tuple-element instructions
    // that use it.
    called_computation()->set_root_instruction(new_root,
                                               /*accept_different_shape=*/true);
    *mutable_shape() = new_root->shape();
    // The instruction might have an existing sharding, which will no longer
    // be valid after we change the shape. So clear the sharding.
    clear_sharding();
    if (root->opcode() == HloOpcode::kTuple) {
      CHECK_OK(called_computation()->RemoveInstruction(root));
    }

    // If this is a newly created multioutput instruction, we need to update
    // the use of the original callable instruction.
    if (newly_created_tuple_instr) {
      HloInstruction* new_instr = AddInstruction(
          HloInstruction::CreateGetTupleElement(root->shape(), this, 0));
      CHECK_OK(ReplaceAllUsesWithDifferentShape(new_instr));
    }
    int64_t index = tuple_elements.size();
    if (do_not_clone) {
      CHECK_EQ(clone, instruction_to_append);
      index -= instruction_to_append->operand_count();
      std::vector<HloInstruction*> to_be_removed;
      const auto& users = instruction_to_append->users();
      to_be_removed.reserve(users.size());
      for (auto old_gte : users) {
        CHECK_EQ(old_gte->opcode(), HloOpcode::kGetTupleElement);
        int64_t old_tuple_index = old_gte->tuple_index();
        HloInstruction* new_gte =
            AddInstruction(HloInstruction::CreateGetTupleElement(
                old_gte->shape(), this, index + old_tuple_index));
        CHECK_OK(old_gte->ReplaceAllUsesWith(new_gte));
        to_be_removed.push_back(old_gte);
      }
      for (auto old_gte : to_be_removed) {
        CHECK_OK(parent()->RemoveInstruction(old_gte));
      }
    } else {
      HloInstruction* new_gte =
          AddInstruction(HloInstruction::CreateGetTupleElement(
              clone->shape(), this, index - 1));
      CHECK_OK(instruction_to_append->ReplaceAllUsesWith(new_gte));
    }
  }

  return clone;
}

absl::InlinedVector<HloComputation*, 1>
HloCallableInstruction::GetOrCloneCalledComputations(
    HloCloneContext* context) const {
  HloModule* module = context != nullptr ? context->module() : GetModule();
  absl::InlinedVector<HloComputation*, 1> new_called_computations;
  for (auto* comp : called_computations()) {
    HloComputation* new_custom_call_computation = nullptr;
    if (context != nullptr) {
      new_custom_call_computation = context->FindComputation(comp);
    }
    if (new_custom_call_computation == nullptr) {
      new_custom_call_computation =
          module->AddEmbeddedComputation(comp->Clone("clone", context));
    }
    new_called_computations.push_back(new_custom_call_computation);
  }
  return new_called_computations;
}

void HloCallableInstruction::RecursivelySetComputationsThreadName(
    std::string_view execution_thread,
    bool skip_async_execution_thread_overwrite) {
  for (HloComputation* comp : called_computations()) {
    SetThreadName(comp, execution_thread,
                  skip_async_execution_thread_overwrite);
  }
}

HloParameterInstruction::HloParameterInstruction(int64_t parameter_number,
                                                 const Shape& shape,
                                                 std::string_view name)
    : HloInstruction(HloOpcode::kParameter, shape),
      parameter_number_(parameter_number) {
  SetAndSanitizeName(name);
}

HloInstructionProto HloParameterInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_parameter_number(parameter_number_);
  if (parameter_replicated_at_leaf_buffers_) {
    for (bool replicated : *parameter_replicated_at_leaf_buffers_) {
      proto.mutable_parameter_replication()->add_replicated_at_leaf_buffers(
          replicated);
    }
  }
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloParameterInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   if (!parameter_replicated_at_leaf_buffers_ || !options.print_ids()) {
//     return;
//   }
//   printer.Next([this](Printer* printer) {
//     printer->Append("parameter_replication={");
//     AppendJoin(printer, *parameter_replicated_at_leaf_buffers_, ",",
//                [](Printer* printer, bool replicated) {
//                  printer->Append(replicated ? "true" : "false");
//                });
//     printer->Append("}");
//   });
// }

void HloParameterInstruction::PrintOperandsWithCanonicalNameMap(
    Printer* printer, const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  if (options.print_parameter_number()) {
    printer->Append(parameter_number_);
  }
}

bool HloParameterInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloParameterInstruction&>(other);
  return parameter_number() == casted_other.parameter_number();
}

std::unique_ptr<HloInstruction>
HloParameterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  auto clone = std::make_unique<HloParameterInstruction>(parameter_number_,
                                                         shape, name());
  if (parameter_replicated_at_leaf_buffers_ &&
      ShapeUtil::Equal(shape, this->shape())) {
    clone->set_parameter_replicated_at_leaf_buffers(
        *parameter_replicated_at_leaf_buffers_);
  }
  return clone;
}

HloGetTupleElementInstruction::HloGetTupleElementInstruction(
    const Shape& shape, HloInstruction* operand, int64_t index)
    : HloInstruction(HloOpcode::kGetTupleElement, shape), tuple_index_(index) {
  AppendOperand(operand);
}

HloInstructionProto HloGetTupleElementInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_tuple_index(tuple_index_);
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloGetTupleElementInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   printer.Next([this](Printer* printer) {
//     AppendCat(printer, "index=", tuple_index());
//   });
// }

bool HloGetTupleElementInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloGetTupleElementInstruction&>(other);
  return tuple_index() == casted_other.tuple_index();
}

std::unique_ptr<HloInstruction>
HloGetTupleElementInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloGetTupleElementInstruction>(shape, new_operands[0],
                                                         tuple_index());
}

HloInfeedInstruction::HloInfeedInstruction(const Shape& infeed_shape,
                                           HloInstruction* token_operand,
                                           const std::string& config)
    : HloInstruction(HloOpcode::kInfeed,
                     ShapeUtil::MakeTupleShape(
                         {infeed_shape, ShapeUtil::MakeTokenShape()})),
      infeed_config_(config) {
  AppendOperand(token_operand);
}

HloInstructionProto HloInfeedInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_infeed_config(infeed_config_);
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloInfeedInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   if (!options.print_infeed_outfeed_config() || infeed_config_.empty()) {
//     return;
//   }
//   printer.Next([this](Printer* printer) {
//     AppendCat(printer, "infeed_config=\"", CEscape(infeed_config_), "\"");
//   });
// }

bool HloInfeedInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  // Not yet supported.
  return false;
}

std::unique_ptr<HloInstruction> HloInfeedInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::make_unique<HloInfeedInstruction>(infeed_shape(), new_operands[0],
                                                infeed_config());
}

HloOutfeedInstruction::HloOutfeedInstruction(const Shape& outfeed_shape,
                                             HloInstruction* operand,
                                             HloInstruction* token_operand,
                                             std::string_view outfeed_config)
    : HloInstruction(HloOpcode::kOutfeed, ShapeUtil::MakeTokenShape()),
      outfeed_shape_(outfeed_shape),
      outfeed_config_(outfeed_config) {
  AppendOperand(operand);
  AppendOperand(token_operand);
}

HloInstructionProto HloOutfeedInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_outfeed_config(outfeed_config());
  *proto.mutable_outfeed_shape() = outfeed_shape().ToProto();
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloOutfeedInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   printer.Next([this](Printer* printer) {
//     printer->Append("outfeed_shape=");
//     ShapeUtil::PrintHumanStringWithLayout(printer, outfeed_shape_);
//   });
//   if (options.print_infeed_outfeed_config() && !outfeed_config_.empty()) {
//     printer.Next([this](Printer* printer) {
//       AppendCat(printer, "outfeed_config=\"", CEscape(outfeed_config_),
//       "\"");
//     });
//   }
// }

bool HloOutfeedInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  // Not yet supported.
  return false;
}

std::unique_ptr<HloInstruction> HloOutfeedInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return std::make_unique<HloOutfeedInstruction>(
      outfeed_shape(), new_operands[0], new_operands[1], outfeed_config());
}

HloDotInstruction::HloDotInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dimension_numbers,
    std::vector<SparsityDescriptor> sparsity,
    absl::Span<HloInstruction* const> sparse_meta)
    : HloInstruction(HloOpcode::kDot, shape),
      dot_dimension_numbers_(dimension_numbers),
      sparsity_(std::move(sparsity)) {
  AppendOperand(lhs);
  AppendOperand(rhs);
  CHECK_LE(sparsity_.size(), kOperands);
  CHECK_EQ(sparsity_.size(), sparse_meta.size());
  for (HloInstruction* meta : sparse_meta) {
    AppendOperand(meta);
  }
  if (sparsity_.size() == kOperands &&
      sparsity_[0].index() > sparsity_[1].index()) {
    std::swap(sparsity_[0], sparsity_[1]);  // Keep descriptors ordered.
    std::swap(mutable_operands()[2], mutable_operands()[3]);
  }
}

HloInstructionProto HloDotInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_dot_dimension_numbers() = dot_dimension_numbers_;
  for (const SparsityDescriptor& descriptor : sparsity_) {
    *proto.add_dot_sparsity() = descriptor;
  }
  return proto;
}

// TODO(chokobole): Uncomment this. Dependency: AttributePrinter
// void HloDotInstruction::PrintExtraAttributesImpl(
//     AttributePrinter& printer, const HloPrintOptions& options) const {
//   printer.Next([this](Printer* printer) {
//     printer->Append(DotDimensionNumbersToString(dot_dimension_numbers_));
//   });
//   if (!sparsity_.empty()) {
//     PrintSparsityDescriptor(printer, absl::MakeSpan(sparsity_));
//   }
// }

bool HloDotInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
        eq_computations) const {
  const auto& casted_other = static_cast<const HloDotInstruction&>(other);
  return protobuf_util::ProtobufEquals(dot_dimension_numbers(),
                                       casted_other.dot_dimension_numbers()) &&
         absl::c_equal(sparsity_, casted_other.sparsity_,
                       protobuf_util::ProtobufEquals);
}

std::unique_ptr<HloInstruction> HloDotInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), kOperands + sparse_operands());
  return std::make_unique<HloDotInstruction>(
      shape, new_operands[0], new_operands[1], dot_dimension_numbers_,
      sparsity_, new_operands.subspan(kOperands));
}

}  //  namespace zkx
