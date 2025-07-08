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

// All HloInstruction subclasses are put in this file.

#ifndef ZKX_HLO_IR_HLO_INSTRUCTIONS_H_
#define ZKX_HLO_IR_HLO_INSTRUCTIONS_H_

#include <stdint.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/base/attributes.h"

#include "xla/tsl/platform/status.h"
#include "zkx/comparison_util.h"
#include "zkx/hlo/ir/collective_device_list.h"
#include "zkx/hlo/ir/hlo_clone_context.h"
#include "zkx/hlo/ir/hlo_computation.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/literal_pool.h"
#include "zkx/shape.h"

namespace zkx {

// Base class for instructions with a dimensions vector.
class HloDimensionsInstruction : public HloInstruction {
 public:
  absl::Span<const int64_t> dimensions() const override { return dimensions_; }

  std::vector<int64_t>* mutable_dimensions() override { return &dimensions_; }

  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    switch (hlo->opcode()) {
      case HloOpcode::kBroadcast:
      case HloOpcode::kConcatenate:
      case HloOpcode::kReduce:
      case HloOpcode::kReverse:
      // TODO(chokobole): Uncomment this. Dependency: HloOpcode::kSort
      // case HloOpcode::kSort:
      case HloOpcode::kTranspose:
        return true;
      default:
        return false;
    }
  }

 protected:
  HloDimensionsInstruction(HloOpcode opcode, const Shape& shape,
                           absl::Span<const int64_t> dimensions)
      : HloInstruction(opcode, shape),
        dimensions_(dimensions.begin(), dimensions.end()) {}
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;

  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  std::vector<int64_t> dimensions_;
};

class HloBroadcastInstruction : public HloDimensionsInstruction {
 public:
  explicit HloBroadcastInstruction(
      const Shape& shape, HloInstruction* operand,
      absl::Span<const int64_t> broadcast_dimension);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kBroadcast;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloFftInstruction : public HloInstruction {
 public:
  explicit HloFftInstruction(const Shape& shape, HloInstruction* operand,
                             FftType fft_type, int64_t fft_length,
                             bool fft_no_bit_reverse);
  FftType fft_type() const { return fft_type_; }

  int64_t fft_length() const { return fft_length_; }

  bool fft_no_bit_reverse() const { return fft_no_bit_reverse_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kFft;
  }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes FFT type for an FFT instruction.
  FftType fft_type_ = FftType::FFT;

  // Indicates the FFT length for an FFT instruction.
  int64_t fft_length_;

  // Indicates whether to apply bit-reverse to the FFT.
  bool fft_no_bit_reverse_ = false;
};

class HloMsmInstruction : public HloInstruction {
 public:
  explicit HloMsmInstruction(const Shape& shape, HloInstruction* scalars,
                             HloInstruction* bases, int32_t window_bits,
                             MsmParallelType msm_parallel_type);
  int32_t window_bits() const { return window_bits_; }
  MsmParallelType msm_parallel_type() const { return msm_parallel_type_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kMsm;
  }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes window bits for an MSM instruction.
  int32_t window_bits_;

  // Describes parallel type for an MSM instruction.
  MsmParallelType msm_parallel_type_;
};

class HloAsyncInstruction : public HloInstruction {
 public:
  // Constructs async-{update,done}.
  HloAsyncInstruction(HloOpcode opcode, const Shape& shape,
                      HloInstruction* operand);

  HloComputation* async_wrapped_computation() const;
  HloInstruction* async_wrapped_instruction() const;
  HloOpcode async_wrapped_opcode() const;

  // Async thread name is a unique thread name for one or more async groups.
  // Typically one HLO module contains a main thread as well as one or more
  // parallel threads.
  virtual std::string_view async_execution_thread() const;
  virtual void set_async_execution_thread(
      std::string_view async_execution_thread) {}
  HloInstructionProto ToProto() const override {
    return HloInstruction::ToProto();
  }

  static bool ClassOf(const HloInstruction* hlo) {
    switch (hlo->opcode()) {
      case HloOpcode::kAsyncStart:
      case HloOpcode::kAsyncUpdate:
      case HloOpcode::kAsyncDone:
        return true;
      default:
        return false;
    }
  }

  // Returns async-start instruction of the async chain.
  HloAsyncInstruction* async_chain_start() const;
  // Returns async-done instruction of the async chain.
  HloAsyncInstruction* async_chain_done() const;
  // Returns the chain of async op referencing this computation,
  // where *begin(GetAsyncChain()) is the async-start op and
  // *end(GetAsyncChain()) is the async-done op.
  std::vector<HloAsyncInstruction*> GetAsyncChain() const;

  bool HasSideEffect() const override {
    return async_wrapped_instruction()->HasSideEffect();
  }

 protected:
  // Helper to constructs async-{start,update,done}.
  HloAsyncInstruction(HloOpcode opcode, const Shape& shape,
                      absl::Span<HloInstruction* const> operands,
                      HloOpcode async_wrapped_opcode);

 private:
  // async-{update,done} inherit all their attributes from async-start,
  // so they shouldn't print any.
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override {
  // }
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  HloAsyncInstruction* async_chain_next_ = nullptr;
};

// Creates async-start.
class HloAsyncStartInstruction : public HloAsyncInstruction {
 public:
  HloAsyncStartInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      HloComputation* async_computation,
      std::string_view async_execution_thread = kMainExecutionThread);

  ~HloAsyncStartInstruction() override;
  void ClearCalledComputations() override;
  // When an async instruction is being destructed, remove it from the vector of
  // pointers of its called computation, to avoid referencing freed memory.
  void ClearAsyncComputationInstruction();

  std::string_view async_execution_thread() const override {
    return async_execution_thread_;
  };
  void set_async_execution_thread(
      std::string_view async_execution_thread) override;
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    switch (hlo->opcode()) {
      case HloOpcode::kAsyncStart:
        return true;
      default:
        return false;
    }
  }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::string async_execution_thread_ = kMainExecutionThread;
};

class HloCopyStartInstruction : public HloInstruction {
 public:
  explicit HloCopyStartInstruction(
      const Shape& shape, HloInstruction* operand,
      std::optional<int> cross_program_prefetch_index);

  std::optional<int> cross_program_prefetch_index() const {
    return cross_program_prefetch_index_;
  }
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kCopyStart;
  }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Each cross program prefetched buffer has a unique index. The indices are
  // assigned contiguously starting from zero in
  // MsaAlgorithm::AllocateCrossProgramPrefetchBuffer. This value is used during
  // codegen to determine which buffer is being speculated at runtime. One
  // possible implementation is to initialize an array with boolean values
  // indicating whether the cross program prefetch succeeds or fails for each
  // buffer.
  std::optional<int> cross_program_prefetch_index_;
};

class HloCompareInstruction : public HloInstruction {
 public:
  explicit HloCompareInstruction(const Shape& shape, HloInstruction* lhs,
                                 HloInstruction* rhs,
                                 ComparisonDirection direction,
                                 std::optional<PrimitiveType> type);
  ComparisonDirection direction() const { return compare_.GetDirection(); }
  ComparisonOrder order() const { return compare_.GetOrder(); }
  PrimitiveType primitive_type() const { return compare_.GetPrimitiveType(); }
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kCompare;
  }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  Comparison compare_;
};

// Class that represents instructions that synchronize and transfer data between
// partitioned devices. Send/Recv and collective instructions (AllReduce,
// AllToAll, CollectivePermute, CollectiveBroadcast) belong to this instruction
// type. A group of instructions (of the same opcode) with the same channel_id
// communicate during execution.
class HloChannelInstruction : public HloInstruction {
 public:
  // Returns the channel id associated with the instruction. The id is
  // shared between each Send/Recv pair or a group of collective instructions
  // and is globally unique to identify each channel.
  std::optional<int64_t> channel_id() const { return channel_id_; }
  void set_channel_id(const std::optional<int64_t>& channel_id);

  // Whether this instruction is identical to `other` except for the values of
  // channel IDs, as long as both have channel IDs or neither has a channel ID.
  virtual bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const {
    return channel_id_.has_value() == other.channel_id().has_value();
  }

  static bool ClassOf(const HloInstruction* hlo);

 protected:
  explicit HloChannelInstruction(HloOpcode opcode, const Shape& shape,
                                 const std::optional<int64_t>& channel_id);

  HloInstructionProto ToProto() const override;
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;

  // Do not override IdenticalSlowPath(). Override
  // IdenticalSlowPathIgnoringChannelIdValues() instead.
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const final;

  std::optional<int64_t> channel_id_;
};

class HloSendRecvInstruction : public HloChannelInstruction {
 public:
  // Returns whether this send/recv instruction sends data to/from the host.
  bool is_host_transfer() const { return is_host_transfer_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    switch (hlo->opcode()) {
      case HloOpcode::kSend:
      case HloOpcode::kSendDone:
      case HloOpcode::kRecv:
      case HloOpcode::kRecvDone:
        return true;
      default:
        return false;
    }
  }

 protected:
  explicit HloSendRecvInstruction(HloOpcode opcode, const Shape& shape,
                                  std::optional<int64_t> channel_id,
                                  bool is_host_transfer);

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Whether this send/recv instruction sends data to/from the host.
  bool is_host_transfer_;
};

class HloSendInstruction : public HloSendRecvInstruction {
 public:
  explicit HloSendInstruction(HloInstruction* operand, HloInstruction* token,
                              std::optional<int64_t> channel_id,
                              bool is_host_transfer);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kSend;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloSendDoneInstruction : public HloSendRecvInstruction {
 public:
  explicit HloSendDoneInstruction(HloSendInstruction* operand,
                                  bool is_host_transfer);
  explicit HloSendDoneInstruction(HloInstruction* operand,
                                  std::optional<int64_t> channel_id,
                                  bool is_host_transfer);
  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kSendDone;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloRecvInstruction : public HloSendRecvInstruction {
 public:
  explicit HloRecvInstruction(const Shape& shape, HloInstruction* token,
                              std::optional<int64_t> channel_id,
                              bool is_host_transfer);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kRecv;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloRecvDoneInstruction : public HloSendRecvInstruction {
 public:
  explicit HloRecvDoneInstruction(HloRecvInstruction* operand,
                                  bool is_host_transfer);
  explicit HloRecvDoneInstruction(HloInstruction* operand,
                                  std::optional<int64_t> channel_id,
                                  bool is_host_transfer);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kRecvDone;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloCollectiveInstruction : public HloChannelInstruction {
 public:
  const std::vector<ReplicaGroup>& replica_groups() const {
    return device_list_.replica_groups();
  }

  const CollectiveDeviceList& device_list() const { return device_list_; }

  // Returns true if the layout of the AllReduce is enforced by XLA client (as
  // the layout set in the shape). The only reason for the client to set the
  // layout is to separately compile computations that communicate with
  // AllReduce. Since this field is only set `true` by the client, the compiler
  // only needs to propagate existing values (e.g., Clone, X64Rewriter) or set
  // `false` for all other cases.
  //
  // When this is `true`, there may be communication endpoints outside the
  // current compilation unit, so the compiler considers this AllReduce as
  // side-effecting to disable compiler transformations. The compiler is free to
  // transform unconstrained AllReduces differently across compilation units.
  // It is an error for an HloModule to have a mix of constrained and
  // unconstrained AllReduce instructions (checked by HloVerifier).
  bool constrain_layout() const { return constrain_layout_; }

  static bool ClassOf(const HloInstruction* hlo);

 protected:
  explicit HloCollectiveInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      const CollectiveDeviceList& collective_device_list, bool constrain_layout,
      const std::optional<int64_t>& channel_id);

  HloInstructionProto ToProto() const override;
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  CollectiveDeviceList device_list_;
  bool constrain_layout_;
};

class HloAllGatherInstruction : public HloCollectiveInstruction {
 public:
  explicit HloAllGatherInstruction(HloOpcode opcode, const Shape& shape,
                                   absl::Span<HloInstruction* const> operands,
                                   int64_t all_gather_dimension,
                                   const CollectiveDeviceList& device_list,
                                   bool constrain_layout,
                                   const std::optional<int64_t>& channel_id,
                                   bool use_global_device_ids);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  explicit HloAllGatherInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands, int64_t all_gather_dimension,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids);

  // Same as HloAllReduceInstruction::use_global_device_ids.
  bool use_global_device_ids() const { return use_global_device_ids_; }

  // The dimension on which data from different participants are concatenated.
  int64_t all_gather_dimension() const { return all_gather_dimension_; }
  absl::Span<const int64_t> dimensions() const override {
    return absl::MakeConstSpan(&all_gather_dimension_, 1);
  }

  void set_all_gather_dimension(int64_t dim) { all_gather_dimension_ = dim; }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kAllGather ||
           hlo->opcode() == HloOpcode::kAllGatherStart;
  }

 protected:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t all_gather_dimension_;
  bool use_global_device_ids_;
};

// Base class for all-reduce and all-reduce scatter instructions.
class HloAllReduceInstructionBase : public HloCollectiveInstruction {
 public:
  explicit HloAllReduceInstructionBase(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      const CollectiveDeviceList& device_list, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids);

  // Returns true if the ids in the ReplicaGroup config represent a global id of
  // (replica_id * partition_count + partition_id) instead of a replica id.
  // This enables more flexible grouping of devices if this all-reduce is both
  // cross-partition and cross-replica.
  //
  // For example with 2 replicas and 4 partitions,
  // replica_groups={{0,1,4,5},{2,3,6,7}}, use_global_device_ids=true means that
  // group[0] = (0,0), (0,1), (1,0), (1,1)
  // group[1] = (0,2), (0,3), (1,2), (1,3)
  // where each pair is (replica_id, partition_id).
  bool use_global_device_ids() const { return use_global_device_ids_; }
  void set_use_global_device_ids(bool value) { use_global_device_ids_ = value; }

  static bool ClassOf(const HloInstruction* hlo);

 protected:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  HloInstructionProto ToProto() const override;

  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

 private:
  bool use_global_device_ids_;
};

class HloAllReduceInstruction : public HloAllReduceInstructionBase {
 public:
  using HloAllReduceInstructionBase::HloAllReduceInstructionBase;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kAllReduce ||
           hlo->opcode() == HloOpcode::kAllReduceStart;
  }

  // Returns true if the AllReduce does no communication, so it's equivalent
  // to a mem copy.
  bool IsNoop() const;

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloReduceScatterInstruction : public HloAllReduceInstructionBase {
 public:
  explicit HloReduceScatterInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      const CollectiveDeviceList& device_list, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids,
      int64_t scatter_dimension);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  explicit HloReduceScatterInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids,
      int64_t scatter_dimension);

  // The dimension on which reduced data is scattered to different participants.
  int64_t scatter_dimension() const { return scatter_dimension_; }
  absl::Span<const int64_t> dimensions() const override {
    return absl::MakeConstSpan(&scatter_dimension_, 1);
  }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kReduceScatter;
  }

 protected:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;

  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t scatter_dimension_;
};

class HloAllToAllInstruction : public HloCollectiveInstruction {
 public:
  explicit HloAllToAllInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      const CollectiveDeviceList& device_list, bool constrain_layout,
      const std::optional<int64_t>& channel_id,
      const std::optional<int64_t>& split_dimension);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  explicit HloAllToAllInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id,
      const std::optional<int64_t>& split_dimension);

  // AllToAll can optionally take a split dimension, which means that this
  // AllToAll takes a single (flattened) array operand and produces an array
  // output (instead of taking a list of operands and producing a tuple).
  //
  // split_dimension specifies which dimension in the operand is split across
  // devices in each replica_group, and also means the concatenated dimension
  // on the output (i.e., input and the output shapes are the same).
  std::optional<int64_t> split_dimension() const { return split_dimension_; }
  void set_split_dimension(int64_t dim) { split_dimension_ = dim; }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kAllToAll;
  }

 protected:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::optional<int64_t> split_dimension_;
};

class HloRaggedAllToAllInstruction : public HloCollectiveInstruction {
 public:
  explicit HloRaggedAllToAllInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      const CollectiveDeviceList& device_list,
      const std::optional<int64_t>& channel_id);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  explicit HloRaggedAllToAllInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      absl::Span<const ReplicaGroup> replica_groups,
      const std::optional<int64_t>& channel_id);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kRaggedAllToAll;
  }

 protected:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;

  HloInstructionProto ToProto() const override;

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloCollectiveBroadcastInstruction : public HloCollectiveInstruction {
 public:
  explicit HloCollectiveBroadcastInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      const CollectiveDeviceList& device_list, bool constrain_layout,
      const std::optional<int64_t>& channel_id);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  explicit HloCollectiveBroadcastInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id);

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kCollectiveBroadcast;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloCollectivePermuteInstruction : public HloChannelInstruction {
 public:
  explicit HloCollectivePermuteInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      const std::optional<int64_t>& channel_id);

  explicit HloCollectivePermuteInstruction(
      HloOpcode opcode, const Shape& shape, HloInstruction* input,
      HloInstruction* output, HloInstruction* input_start_indices,
      HloInstruction* output_start_indices,
      absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
      absl::Span<const std::vector<int64_t>> slice_sizes,
      const std::optional<int64_t>& channel_id);

  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs() const {
    return source_target_pairs_;
  }

  const std::vector<std::vector<int64_t>>& dynamic_slice_sizes_list() const {
    return slice_sizes_;
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kCollectivePermute ||
           hlo->opcode() == HloOpcode::kCollectivePermuteStart;
  }

  bool inplace() const { return inplace_; }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  const std::vector<std::pair<int64_t, int64_t>> source_target_pairs_;
  const std::vector<std::vector<int64_t>> slice_sizes_;
  bool inplace_;
};

inline bool HloAllReduceInstructionBase::ClassOf(const HloInstruction* hlo) {
  return HloAllReduceInstruction::ClassOf(hlo) ||
         hlo->opcode() == HloOpcode::kReduceScatter;
}

inline bool HloCollectiveInstruction::ClassOf(const HloInstruction* hlo) {
  return HloAllReduceInstructionBase::ClassOf(hlo) ||
         HloCollectiveBroadcastInstruction::ClassOf(hlo) ||
         HloAllGatherInstruction::ClassOf(hlo) ||
         HloAllToAllInstruction::ClassOf(hlo) ||
         HloRaggedAllToAllInstruction::ClassOf(hlo);
}

inline bool HloChannelInstruction::ClassOf(const HloInstruction* hlo) {
  return HloCollectiveInstruction::ClassOf(hlo) ||
         HloCollectivePermuteInstruction::ClassOf(hlo) ||
         HloSendRecvInstruction::ClassOf(hlo);
}

class HloReverseInstruction : public HloDimensionsInstruction {
 public:
  explicit HloReverseInstruction(const Shape& shape, HloInstruction* operand,
                                 absl::Span<const int64_t> dimensions);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kReverse;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloConcatenateInstruction : public HloDimensionsInstruction {
 public:
  explicit HloConcatenateInstruction(const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     int64_t dimension);
  // Accessor for the dimension in which a concatenate HLO should occur.
  int64_t concatenate_dimension() const override {
    return HloInstruction::dimensions(0);
  }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kConcatenate;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloSliceInstruction : public HloInstruction {
 public:
  explicit HloSliceInstruction(const Shape& shape, HloInstruction* operand,
                               absl::Span<const int64_t> start_indices,
                               absl::Span<const int64_t> limit_indices,
                               absl::Span<const int64_t> strides);

  HloInstructionProto ToProto() const override;

  // Returns the start index in the given dimension for a slice node.
  int64_t slice_starts(int64_t dimension) const {
    return slice_starts_[dimension];
  }
  const std::vector<int64_t>& slice_starts() const { return slice_starts_; }
  std::vector<int64_t>* mutable_slice_starts() { return &slice_starts_; }

  // Returns the (exclusive) limit index in the given dimension for a slice
  // node.
  int64_t slice_limits(int64_t dimension) const {
    return slice_limits_[dimension];
  }
  const std::vector<int64_t>& slice_limits() const { return slice_limits_; }
  std::vector<int64_t>* mutable_slice_limits() { return &slice_limits_; }

  // Returns the stride in the given dimension for a slice node.
  int64_t slice_strides(int64_t dimension) const {
    return slice_strides_[dimension];
  }
  const std::vector<int64_t>& slice_strides() const { return slice_strides_; }
  std::vector<int64_t>* mutable_slice_strides() { return &slice_strides_; }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kSlice;
  }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes the [begin, end) index range for a slice.
  std::vector<int64_t> slice_starts_;
  std::vector<int64_t> slice_limits_;
  std::vector<int64_t> slice_strides_;
};

class HloConstantInstruction : public HloInstruction {
 public:
  explicit HloConstantInstruction(Literal literal);
  HloConstantInstruction(Literal literal, const Shape& shape);
  HloConstantInstruction(std::shared_ptr<Literal> literal, const Shape& shape);
  // Used when the literal is too large and dropped.
  explicit HloConstantInstruction(const Shape& shape);
  // Returns the literal associated with this instruction.
  const Literal& literal() const { return *literal_; }
  // Returns the (mutable) literal associated with this instruction.
  // Clone the literal if necessary (do not modify the shared instance).
  Literal* mutable_literal() {
    if (literal_.use_count() > 1) {
      literal_.reset(new Literal(literal_->Clone()));
    }
    return literal_.get();
  }
  // Returns whether there is literal associated with this instruction.
  bool HasLiteral() const { return static_cast<bool>(literal_); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kConstant;
  }

  // Canonicalize constant literal using the given literal pool.
  bool Canonicalize(LiteralPool* literal_pool) {
    if (literal_pool && literal_) {
      auto canonical = literal_pool->GetCanonicalLiteral(literal_);
      if (canonical != literal_) {
        literal_ = std::move(canonical);
        return true;
      }
    }
    return false;
  }

  // Add literal to the hash state.
  void HashAdditionalAttributes(absl::HashState h) const override {
    if (HasLiteral()) {
      absl::HashState::combine(std::move(h),
                               Literal::AbslHashable<true>(literal()));
    }
  }

 private:
  bool IsElementwiseImpl(
      const std::optional<int64_t>& operand_idx) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  void PrintOperandsWithCanonicalNameMap(
      Printer* printer, const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  std::shared_ptr<Literal> literal_;
};

class HloParameterInstruction : public HloInstruction {
 public:
  explicit HloParameterInstruction(int64_t parameter_number, const Shape& shape,
                                   std::string_view name);
  int64_t parameter_number() const { return parameter_number_; }

  // Sets and gets the whether all replicas will receive the same parameter data
  // for each leaf buffer in data parallelism.
  void set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool> parameter_replicated_at_leaf_buffers) {
    CHECK_EQ(ShapeUtil::GetLeafCount(shape()),
             parameter_replicated_at_leaf_buffers.size());
    parameter_replicated_at_leaf_buffers_.emplace(
        parameter_replicated_at_leaf_buffers.begin(),
        parameter_replicated_at_leaf_buffers.end());
  }
  void set_parameter_replicated_at_leaf_buffers(
      const std::vector<bool>& parameter_replicated_at_leaf_buffers) {
    CHECK_EQ(ShapeUtil::GetLeafCount(shape()),
             parameter_replicated_at_leaf_buffers.size());
    parameter_replicated_at_leaf_buffers_ =
        parameter_replicated_at_leaf_buffers;
  }
  const std::optional<std::vector<bool>>& parameter_replicated_at_leaf_buffers()
      const {
    return parameter_replicated_at_leaf_buffers_;
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kParameter;
  }

  // Add parameter number to the hash.
  void HashAdditionalAttributes(absl::HashState h) const override {
    absl::HashState::combine(std::move(h), parameter_number());
  }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  void PrintOperandsWithCanonicalNameMap(
      Printer* printer, const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t parameter_number_ = 0;

  // Specifies whether each buffer has the same parameter value on all replicas
  // in data parallelism.
  std::optional<std::vector<bool>> parameter_replicated_at_leaf_buffers_;
};

class HloGetTupleElementInstruction : public HloInstruction {
 public:
  explicit HloGetTupleElementInstruction(const Shape& shape,
                                         HloInstruction* operand,
                                         int64_t index);
  // Returns the tuple index associated with this instruction.
  int64_t tuple_index() const { return tuple_index_; }
  // Sets the tuple index associated with this instruction.
  void set_tuple_index(int64_t new_tuple_index) {
    tuple_index_ = new_tuple_index;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kGetTupleElement;
  }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t tuple_index_ = -1;
};

class HloInfeedInstruction : public HloInstruction {
 public:
  explicit HloInfeedInstruction(const Shape& infeed_shape,
                                HloInstruction* token_operand,
                                const std::string& config);
  // Returns the infeed configuration string. The infeed configuration includes
  // any metadata needed for the backend compiler (e.g., infeed buffer address)
  // and is target-dependent.
  std::string infeed_config() const { return infeed_config_; }
  void set_infeed_config(const std::string& config) { infeed_config_ = config; }
  // Returns the shape of the data received by the infeed. This is not the same
  // as the shape of the infeed instruction which produces a tuple containing
  // the infeed data shape and a TOKEN.
  const Shape& infeed_shape() const {
    TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape()));
    return ShapeUtil::GetSubshape(shape(), {0});
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kInfeed;
  }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The string representation of the infeed configuration.
  std::string infeed_config_;
};

class HloOutfeedInstruction : public HloInstruction {
 public:
  explicit HloOutfeedInstruction(const Shape& outfeed_shape,
                                 HloInstruction* operand,
                                 HloInstruction* token_operand,
                                 std::string_view outfeed_config);
  // Returns the shape for the Outfeed instruction.
  const Shape& outfeed_shape() const { return outfeed_shape_; }
  // Returns the mutable shape for the Outfeed instruction.
  Shape* mutable_outfeed_shape() { return &outfeed_shape_; }
  // Returns the config for the Outfeed instruction.
  const std::string& outfeed_config() const { return outfeed_config_; }
  void set_outfeed_config(const std::string& config) {
    outfeed_config_ = config;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kOutfeed;
  }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Shape of outfeed request.
  Shape outfeed_shape_;
  // Outfeed configuration information, only present for kOutfeed.
  std::string outfeed_config_;
};

class HloDotInstruction : public HloInstruction {
 public:
  static const int kOperands = 2;

  // Creates a dot op with operands `lhs` and `rhs` with contracting and batch
  // dimensions specified in `dimension_numbers`. If `sparsity` is set, then
  // `sparse_meta` must also be present (and have the same size).
  explicit HloDotInstruction(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      const DotDimensionNumbers& dimension_numbers,
      std::vector<SparsityDescriptor> sparsity = {},
      absl::Span<HloInstruction* const> sparse_meta = {});

  // Returns data on the dimension numbers used for a dot operation.
  const DotDimensionNumbers& dot_dimension_numbers() const {
    return dot_dimension_numbers_;
  }

  // Sets dimension numbers used for a dot operation.
  DotDimensionNumbers* mutable_dot_dimension_numbers() {
    return &dot_dimension_numbers_;
  }

  // Sparsity descriptors are optional. If present, additional operands define
  // how the data is read for the dot inputs.
  int sparse_operands() const { return sparsity_.size(); }
  absl::Span<const SparsityDescriptor> sparsity() const {
    return absl::MakeSpan(sparsity_);
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kDot;
  }

 private:
  // TODO(chokobole): Uncomment this. Dependency: AttributePrinter
  // void PrintExtraAttributesImpl(AttributePrinter& printer,
  //                               const HloPrintOptions& options) const
  //                               override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes the dimension numbers used for a dot.
  DotDimensionNumbers dot_dimension_numbers_;

  // Sparsity descriptors are set if some operands are sparse. In this case, the
  // additional metadata operands contain the information that defines how
  // the data is read.
  std::vector<SparsityDescriptor> sparsity_;
};

}  // namespace zkx

#endif  // ZKX_HLO_IR_HLO_INSTRUCTIONS_H_
