/* Copyright 2025 The OpenXLA Authors.

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

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/cpu/runtime/all_gather_thunk.h"
#include "zkx/backends/cpu/runtime/all_reduce_thunk.h"
#include "zkx/backends/cpu/runtime/all_to_all_thunk.h"
#include "zkx/backends/cpu/runtime/collective_permute_thunk.h"
#include "zkx/backends/cpu/runtime/collective_thunk.h"
#include "zkx/backends/cpu/runtime/conditional_thunk.h"
#include "zkx/backends/cpu/runtime/copy_thunk.h"
#include "zkx/backends/cpu/runtime/infeed_thunk.h"
#include "zkx/backends/cpu/runtime/kernel_thunk.h"
#include "zkx/backends/cpu/runtime/outfeed_thunk.h"
#include "zkx/backends/cpu/runtime/reduce_scatter_thunk.h"
#include "zkx/backends/cpu/runtime/resource_use.h"
#include "zkx/backends/cpu/runtime/serdes_base.h"
#include "zkx/backends/cpu/runtime/thunk.h"
#include "zkx/backends/cpu/runtime/thunk_executor.h"
#include "zkx/backends/cpu/runtime/thunk_proto_serdes.h"
#include "zkx/backends/cpu/runtime/thunk_testlib.h"
#include "zkx/backends/cpu/runtime/while_thunk.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/service/collective_ops_utils.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/stream_executor/launch_dim.h"
#include "zkx/zkx_data.pb.h"

namespace zkx::cpu {
namespace {

// Thunk sequence serdes test base.
// This is independent of the serialization format.
template <typename T>
class ThunkSequenceSerdesTest : public ::testing::Test {
 protected:
  explicit ThunkSequenceSerdesTest() = default;

  absl::StatusOr<ThunkSequence> CreateThunkSequenceFromAllThunkTypes() {
    // NOTE create buffer allocations using thunk_testlib
    ThunkSequence thunk_sequence;

    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateAllGatherThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateAllReduceThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateAllToAllThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(),
                        CreateReduceScatterThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(),
                        CreateCollectivePermuteThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateCopyThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(),
                        CreateConditionalThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateInfeedThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateOutfeedThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateWhileThunk());
    TF_ASSIGN_OR_RETURN(thunk_sequence.emplace_back(), CreateKernelThunk());
    return thunk_sequence;
  }

  absl::StatusOr<std::string> Serialize(const ThunkSequence& thunk_sequence) {
    return thunk_sequence_serdes_->Serialize(thunk_sequence);
  }

  absl::StatusOr<std::unique_ptr<ThunkSequence>> Deserialize(
      const std::string& serialized) {
    return thunk_sequence_serdes_->Deserialize(serialized);
  }

  bool VerifyThunkSequenceEquality(const ThunkSequence& thunk_sequence_1,
                                   const ThunkSequence& thunk_sequence_2) {
    if (thunk_sequence_1.size() != thunk_sequence_2.size()) {
      return false;
    }
    for (int i = 0; i < thunk_sequence_1.size(); ++i) {
      if (!VerifyThunkEquality(*thunk_sequence_1[i], *thunk_sequence_2[i])) {
        return false;
      }
    }
    return true;
  }

 public:
  void SetUp() override {
    // HACK(basioli): allocations are created on thunk creation and are pushed
    // back into this vector. If we don't reserve enough space, reallocation
    // will get triggered which will invalidate the pointers to the allocations
    // owned by the thunks.
    buffer_allocations_.reserve(10000);
    thunk_sequence_serdes_ = std::make_unique<T>(&buffer_allocations_);
  }

 private:
  void AddBufferAllocations(const size_t no_of_allocations_to_add) {
    for (size_t i = 0; i < no_of_allocations_to_add; ++i) {
      literals_.push_back(LiteralUtil::CreateFull<int32_t>({2, 4}, 0));
      buffer_allocations_.push_back(
          CreateBufferAllocation(buffer_allocations_.size(), literals_.back()));
    }
  }
  // Thunk creation helper functions.
  absl::StatusOr<std::unique_ptr<Thunk>> CreateAllGatherThunk() {
    AddBufferAllocations(2);

    return AllGatherThunk::Create(
        Thunk::Info(),
        /*op_params=*/
        {/*op_id=*/0,
         /*has_channel_id=*/false,
         /*use_global_device_ids=*/false,
         /*group=*/{}},
        /*op_buffers=*/
        {
            /*source_buffers=*/{CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 2])},
            /*source_shapes=*/
            {literals_[buffer_allocations_.size() - 2].shape()},
            /*destination_buffers=*/
            {CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 1])},
            /*destination_shapes=*/
            {literals_[buffer_allocations_.size() - 1].shape()},
        },
        /*op_resources=*/
        {
            /*communicator_resource=*/nullptr,
        });
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateAllReduceThunk() {
    AddBufferAllocations(2);

    return AllReduceThunk::Create(
        Thunk::Info(), ReductionKind::kSum,
        /*op_params=*/
        {/*op_id=*/0,
         /*has_channel_id=*/false,
         /*use_global_device_ids=*/false,
         /*group=*/{}},
        /*op_buffers=*/
        {
            /*source_buffers=*/{CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 2])},
            /*source_shapes=*/
            {literals_[buffer_allocations_.size() - 2].shape()},
            /*destination_buffers=*/
            {CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 1])},
            /*destination_shapes=*/
            {literals_[buffer_allocations_.size() - 1].shape()},
        },
        /*op_resources=*/
        {
            /*communicator_resource=*/nullptr,
        },
        /*single_replica=*/false);
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateAllToAllThunk() {
    AddBufferAllocations(2);

    return AllToAllThunk::Create(
        Thunk::Info(),
        /*op_params=*/
        {/*op_id=*/0,
         /*has_channel_id=*/false,
         /*use_global_device_ids=*/false,
         /*group=*/{}},
        /*op_buffers=*/
        {
            /*source_buffers=*/{CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 2])},
            /*source_shapes=*/
            {literals_[buffer_allocations_.size() - 2].shape()},
            /*destination_buffers=*/
            {CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 1])},
            /*destination_shapes=*/
            {literals_[buffer_allocations_.size() - 1].shape()},
        },
        /*op_resources=*/
        {
            /*communicator_resource=*/nullptr,
        });
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateReduceScatterThunk() {
    AddBufferAllocations(2);

    return ReduceScatterThunk::Create(
        Thunk::Info(), ReductionKind::kSum,
        /*op_params=*/
        {/*op_id=*/0,
         /*has_channel_id=*/false,
         /*use_global_device_ids=*/false,
         /*group=*/{}},
        /*op_buffers=*/
        {
            /*source_buffers=*/{CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 2])},
            /*source_shapes=*/
            {literals_[buffer_allocations_.size() - 2].shape()},
            /*destination_buffers=*/
            {CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 1])},
            /*destination_shapes=*/
            {literals_[buffer_allocations_.size() - 1].shape()},
        },
        /*op_resources=*/
        {
            /*communicator_resource=*/nullptr,
        });
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateCollectivePermuteThunk() {
    AddBufferAllocations(2);

    return CollectivePermuteThunk::Create(
        Thunk::Info(),
        /*op_params=*/
        {/*op_id=*/0,
         /*has_channel_id=*/false,
         /*use_global_device_ids=*/false,
         /*group=*/{}},
        /*op_buffers=*/
        {
            /*source_buffers=*/{CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 2])},
            /*source_shapes=*/
            {literals_[buffer_allocations_.size() - 2].shape()},
            /*destination_buffers=*/
            {CreateBufferAllocationSlice(
                buffer_allocations_[buffer_allocations_.size() - 1])},
            /*destination_shapes=*/
            {literals_[buffer_allocations_.size() - 1].shape()},
        },
        /*op_resources=*/
        {
            /*communicator_resource=*/nullptr,
        },
        {{0, 0}});
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateCopyThunk() {
    AddBufferAllocations(2);

    return CopyThunk::Create(
        Thunk::Info(),
        /*src_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 2]),
        /*src_shape=*/literals_[buffer_allocations_.size() - 2].shape(),
        /*dst_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        /*dst_shape=*/literals_[buffer_allocations_.size() - 1].shape());
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateConditionalThunk() {
    std::vector<ThunkSequence> branch_sequences;
    for (int i = 0; i < 2; ++i) {
      ThunkSequence called_sequence;
      TF_ASSIGN_OR_RETURN(called_sequence.emplace_back(),
                          CreateAllGatherThunk());
      TF_ASSIGN_OR_RETURN(called_sequence.emplace_back(),
                          CreateAllReduceThunk());
      TF_ASSIGN_OR_RETURN(called_sequence.emplace_back(),
                          CreateAllToAllThunk());
      branch_sequences.push_back(std::move(called_sequence));
    }

    AddBufferAllocations(1);

    return ConditionalThunk::Create(
        Thunk::Info(),
        /*branch_index_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        std::move(branch_sequences));
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateInfeedThunk() {
    AddBufferAllocations(2);

    return InfeedThunk::Create(
        Thunk::Info(),
        /*infeed_buffers=*/
        {{
             CreateBufferAllocationSlice(
                 buffer_allocations_[buffer_allocations_.size() - 2]),
             literals_[buffer_allocations_.size() - 2].shape(),
         },
         {
             CreateBufferAllocationSlice(
                 buffer_allocations_[buffer_allocations_.size() - 1]),
             literals_[buffer_allocations_.size() - 1].shape(),
         }},
        InfeedThunk::InfeedResources());
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateOutfeedThunk() {
    AddBufferAllocations(2);

    return OutfeedThunk::Create(
        Thunk::Info(),
        /*outfeed_buffers=*/
        {{
             CreateBufferAllocationSlice(
                 buffer_allocations_[buffer_allocations_.size() - 2]),
             literals_[buffer_allocations_.size() - 2].shape(),
         },
         {
             CreateBufferAllocationSlice(
                 buffer_allocations_[buffer_allocations_.size() - 1]),
             literals_[buffer_allocations_.size() - 1].shape(),
         }},
        OutfeedThunk::OutfeedResources());
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateWhileThunk() {
    ThunkSequence cond_sequence;
    TF_ASSIGN_OR_RETURN(cond_sequence.emplace_back(), CreateAllGatherThunk());
    ThunkSequence body_sequence;
    TF_ASSIGN_OR_RETURN(body_sequence.emplace_back(), CreateAllGatherThunk());
    TF_ASSIGN_OR_RETURN(body_sequence.emplace_back(), CreateAllReduceThunk());
    TF_ASSIGN_OR_RETURN(body_sequence.emplace_back(), CreateAllToAllThunk());

    AddBufferAllocations(1);
    return WhileThunk::Create(
        Thunk::Info(),
        /*cond_buffer=*/
        CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1]),
        /*cond_sequence=*/std::move(cond_sequence),
        /*body_sequence=*/std::move(body_sequence),
        /*trip_count=*/1);
  }

  absl::StatusOr<std::unique_ptr<Thunk>> CreateKernelThunk() {
    AddBufferAllocations(2);
    return KernelThunk::Create(
        Thunk::Info(),
        /*arguments_buffers=*/
        {CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 2])},
        /*results_buffers=*/
        {CreateBufferAllocationSlice(
            buffer_allocations_[buffer_allocations_.size() - 1])},
        /*kernel_name=*/"test",
        /*thread_dim=*/se::ThreadDim(1),
        /*invariant_arguments=*/{{0}},
        /*min_alignment=*/8);
  }

  bool VerifySliceEquality(const BufferAllocation::Slice& slice_1,
                           const BufferAllocation::Slice& slice_2) {
    return slice_1.offset() == slice_2.offset() &&
           slice_1.size() == slice_2.size() &&
           slice_1.allocation() == slice_2.allocation();
  }

  bool VerifySliceShapeEquality(const BufferAllocation::Slice& slice_1,
                                const Shape& shape_1,
                                const BufferAllocation::Slice& slice_2,
                                const Shape& shape_2) {
    return VerifySliceEquality(slice_1, slice_2) &&
           ShapeUtil::Equal(shape_1, shape_2);
  }

  bool VerifyShapesEquality(absl::Span<const Shape> shapes_1,
                            absl::Span<const Shape> shapes_2) {
    if (shapes_1.size() != shapes_2.size()) {
      return false;
    }

    for (size_t i = 0; i < shapes_1.size(); ++i) {
      if (!ShapeUtil::Equal(shapes_1[i], shapes_2[i])) {
        return false;
      }
    }
    return true;
  }

  bool VerifySlicesEquality(
      absl::Span<const BufferAllocation::Slice> slices_1,
      absl::Span<const BufferAllocation::Slice> slices_2) {
    if (slices_1.size() != slices_2.size()) {
      return false;
    }

    for (size_t i = 0; i < slices_1.size(); ++i) {
      if (!VerifySliceEquality(slices_1[i], slices_2[i])) {
        return false;
      }
    }
    return true;
  }

  bool VerifyResourceEquality(const std::shared_ptr<Resource>& resource_1,
                              const std::shared_ptr<Resource>& resource_2) {
    if ((resource_1 == nullptr) ^ (resource_2 == nullptr)) {
      return false;
    }

    if (resource_1 && resource_1->kind() != resource_2->kind()) {
      return false;
    }

    return true;
  }

  bool VerifyCollectiveThunkEquality(const CollectiveThunk& thunk_1,
                                     const CollectiveThunk& thunk_2) {
    const auto& op_params_1 = thunk_1.op_params();
    const auto& op_params_2 = thunk_2.op_params();

    bool are_replica_groups_equal = absl::c_equal(
        op_params_1.group, op_params_2.group,
        [](const ReplicaGroup& group_1, const ReplicaGroup& group_2) {
          return absl::c_equal(group_1.replica_ids(), group_2.replica_ids());
        });

    if (op_params_1.op_id != op_params_2.op_id ||
        op_params_1.has_channel_id != op_params_2.has_channel_id ||
        op_params_1.use_global_device_ids !=
            op_params_2.use_global_device_ids ||
        !are_replica_groups_equal) {
      return false;
    }

    const auto& op_buffers_1 = thunk_1.op_buffers();
    const auto& op_buffers_2 = thunk_2.op_buffers();

    if (!VerifySlicesEquality(op_buffers_1.source_buffers,
                              op_buffers_2.source_buffers) ||
        !VerifySlicesEquality(op_buffers_1.destination_buffers,
                              op_buffers_2.destination_buffers) ||
        !VerifyShapesEquality(op_buffers_1.source_shapes,
                              op_buffers_2.source_shapes) ||
        !VerifyShapesEquality(op_buffers_1.destination_shapes,
                              op_buffers_2.destination_shapes)) {
      return false;
    }

    const auto& op_resources_1 = thunk_1.op_resources();
    const auto& op_resources_2 = thunk_2.op_resources();

    if (!VerifyResourceEquality(op_resources_1.communicator_resource,
                                op_resources_2.communicator_resource)) {
      return false;
    }

    return true;
  }

  bool VerifyAllGatherThunkEquality(const AllGatherThunk& thunk_1,
                                    const AllGatherThunk& thunk_2) {
    return VerifyCollectiveThunkEquality(thunk_1, thunk_2);
  }

  bool VerifyAllReduceThunkEquality(const AllReduceThunk& thunk_1,
                                    const AllReduceThunk& thunk_2) {
    return thunk_1.single_replica() == thunk_2.single_replica() &&
           thunk_1.reduction_kind() == thunk_2.reduction_kind() &&
           VerifyCollectiveThunkEquality(thunk_1, thunk_2);
  }

  bool VerifyAllToAllThunkEquality(const AllToAllThunk& thunk_1,
                                   const AllToAllThunk& thunk_2) {
    return VerifyCollectiveThunkEquality(thunk_1, thunk_2);
  }

  bool VerifyCollectivePermuteThunkEquality(
      const CollectivePermuteThunk& thunk_1,
      const CollectivePermuteThunk& thunk_2) {
    return absl::c_equal(thunk_1.source_target_pairs(),
                         thunk_2.source_target_pairs()) &&
           VerifyCollectiveThunkEquality(thunk_1, thunk_2);
  }

  bool VerifyCopyThunkEquality(const CopyThunk& thunk_1,
                               const CopyThunk& thunk_2) {
    return VerifySliceShapeEquality(thunk_1.src_buffer(), thunk_1.src_shape(),
                                    thunk_2.src_buffer(),
                                    thunk_2.src_shape()) &&
           VerifySliceShapeEquality(thunk_1.dst_buffer(), thunk_1.dst_shape(),
                                    thunk_2.dst_buffer(), thunk_2.dst_shape());
  }

  bool VerifyConditionalThunkEquality(const ConditionalThunk& thunk_1,
                                      const ConditionalThunk& thunk_2) {
    return VerifySliceEquality(thunk_1.branch_index_buffer(),
                               thunk_2.branch_index_buffer()) &&
           absl::c_equal(thunk_1.branch_executors(), thunk_2.branch_executors(),
                         [this](const ThunkExecutor& executor_1,
                                const ThunkExecutor& executor_2) {
                           return absl::c_equal(
                               executor_1.thunk_sequence(),
                               executor_2.thunk_sequence(),
                               [this](const std::unique_ptr<Thunk>& thunk_1,
                                      const std::unique_ptr<Thunk>& thunk_2) {
                                 return VerifyThunkEquality(*thunk_1, *thunk_2);
                               });
                         });
  }

  bool VerifyInfeedThunkEquality(const InfeedThunk& thunk_1,
                                 const InfeedThunk& thunk_2) {
    InfeedThunk::InfeedResources infeed_resources_1 =
        thunk_1.infeed_resources();
    InfeedThunk::InfeedResources infeed_resources_2 =
        thunk_2.infeed_resources();

    if (!VerifyResourceEquality(infeed_resources_1.consume_token,
                                infeed_resources_2.consume_token) ||
        !VerifyResourceEquality(infeed_resources_1.produce_token,
                                infeed_resources_2.produce_token)) {
      return false;
    }

    return absl::c_equal(thunk_1.infeed_buffers(), thunk_2.infeed_buffers(),
                         [this](const InfeedThunk::InfeedBuffer& buffer_1,
                                const InfeedThunk::InfeedBuffer& buffer_2) {
                           return VerifySliceShapeEquality(
                               buffer_1.slice, buffer_1.shape, buffer_2.slice,
                               buffer_2.shape);
                         });
  }

  bool VerifyOutfeedThunkEquality(const OutfeedThunk& thunk_1,
                                  const OutfeedThunk& thunk_2) {
    OutfeedThunk::OutfeedResources outfeed_resources_1 =
        thunk_1.outfeed_resources();
    OutfeedThunk::OutfeedResources outfeed_resources_2 =
        thunk_2.outfeed_resources();

    if (!VerifyResourceEquality(outfeed_resources_1.consume_token,
                                outfeed_resources_2.consume_token) ||
        !VerifyResourceEquality(outfeed_resources_1.produce_token,
                                outfeed_resources_2.produce_token)) {
      return false;
    }

    return absl::c_equal(thunk_1.outfeed_buffers(), thunk_2.outfeed_buffers(),
                         [this](const OutfeedThunk::OutfeedBuffer& buffer_1,
                                const OutfeedThunk::OutfeedBuffer& buffer_2) {
                           return VerifySliceShapeEquality(
                               buffer_1.slice, buffer_1.shape, buffer_2.slice,
                               buffer_2.shape);
                         });
  }

  bool VerifyWhileThunkEquality(const WhileThunk& thunk_1,
                                const WhileThunk& thunk_2) {
    return VerifySliceEquality(thunk_1.cond_buffer(), thunk_2.cond_buffer()) &&
           absl::c_equal(thunk_1.cond_executor().thunk_sequence(),
                         thunk_2.cond_executor().thunk_sequence(),
                         [this](const std::unique_ptr<Thunk>& thunk_1,
                                const std::unique_ptr<Thunk>& thunk_2) {
                           return VerifyThunkEquality(*thunk_1, *thunk_2);
                         }) &&
           absl::c_equal(thunk_1.body_executor().thunk_sequence(),
                         thunk_2.body_executor().thunk_sequence(),
                         [this](const std::unique_ptr<Thunk>& thunk_1,
                                const std::unique_ptr<Thunk>& thunk_2) {
                           return VerifyThunkEquality(*thunk_1, *thunk_2);
                         }) &&
           thunk_1.trip_count() == thunk_2.trip_count();
  }

  bool VerifyKernelThunkEquality(const KernelThunkBase& thunk_1,
                                 const KernelThunkBase& thunk_2) {
    return thunk_1.kernel_name() == thunk_2.kernel_name() &&
           thunk_1.thread_dim() == thunk_2.thread_dim() &&
           thunk_1.min_alignment() == thunk_2.min_alignment() &&
           absl::c_equal(thunk_1.arguments_buffers(),
                         thunk_2.arguments_buffers(),
                         [this](const BufferAllocation::Slice& slice_1,
                                const BufferAllocation::Slice& slice_2) {
                           return VerifySliceEquality(slice_1, slice_2);
                         }) &&
           absl::c_equal(thunk_1.results_buffers(), thunk_2.results_buffers(),
                         [this](const BufferAllocation::Slice& slice_1,
                                const BufferAllocation::Slice& slice_2) {
                           return VerifySliceEquality(slice_1, slice_2);
                         });
  }

  bool VerifyReduceScatterThunkEquality(const ReduceScatterThunk& thunk_1,
                                        const ReduceScatterThunk& thunk_2) {
    return thunk_1.reduction_kind() == thunk_2.reduction_kind() &&
           VerifyCollectiveThunkEquality(thunk_1, thunk_2);
  }

  bool VerifyThunkEquality(const Thunk& thunk_1, const Thunk& thunk_2) {
    if (thunk_1.kind() != thunk_2.kind()) {
      return false;
    }

    if (!(thunk_1.info().op_name == thunk_2.info().op_name &&
          thunk_1.info().module_name == thunk_2.info().module_name &&
          thunk_1.info().module_id == thunk_2.info().module_id)) {
      return false;
    }

    switch (thunk_1.kind()) {
      case Thunk::Kind::kAllGather:
        return VerifyAllGatherThunkEquality(
            static_cast<const AllGatherThunk&>(thunk_1),
            static_cast<const AllGatherThunk&>(thunk_2));
      case Thunk::Kind::kAllReduce:
        return VerifyAllReduceThunkEquality(
            static_cast<const AllReduceThunk&>(thunk_1),
            static_cast<const AllReduceThunk&>(thunk_2));
      case Thunk::Kind::kAllToAll:
        return VerifyAllToAllThunkEquality(
            static_cast<const AllToAllThunk&>(thunk_1),
            static_cast<const AllToAllThunk&>(thunk_2));
      case Thunk::Kind::kCollectivePermute:
        return VerifyCollectivePermuteThunkEquality(
            static_cast<const CollectivePermuteThunk&>(thunk_1),
            static_cast<const CollectivePermuteThunk&>(thunk_2));
      case Thunk::Kind::kCopy:
        return VerifyCopyThunkEquality(static_cast<const CopyThunk&>(thunk_1),
                                       static_cast<const CopyThunk&>(thunk_2));
      case Thunk::Kind::kConditional:
        return VerifyConditionalThunkEquality(
            static_cast<const ConditionalThunk&>(thunk_1),
            static_cast<const ConditionalThunk&>(thunk_2));
      case Thunk::Kind::kInfeed:
        return VerifyInfeedThunkEquality(
            static_cast<const InfeedThunk&>(thunk_1),
            static_cast<const InfeedThunk&>(thunk_2));
      case Thunk::Kind::kOutfeed:
        return VerifyOutfeedThunkEquality(
            static_cast<const OutfeedThunk&>(thunk_1),
            static_cast<const OutfeedThunk&>(thunk_2));
      case Thunk::Kind::kWhile:
        return VerifyWhileThunkEquality(
            static_cast<const WhileThunk&>(thunk_1),
            static_cast<const WhileThunk&>(thunk_2));
      case Thunk::Kind::kKernel:
        return VerifyKernelThunkEquality(
            static_cast<const KernelThunkBase&>(thunk_1),
            static_cast<const KernelThunkBase&>(thunk_2));
      case Thunk::Kind::kReduceScatter:
        return VerifyReduceScatterThunkEquality(
            static_cast<const ReduceScatterThunk&>(thunk_1),
            static_cast<const ReduceScatterThunk&>(thunk_2));
      case Thunk::Kind::kUnknown:
        return false;
    }

    return true;
  }
  std::unique_ptr<SerDesBase<ThunkSequence>> thunk_sequence_serdes_;
  std::vector<BufferAllocation> buffer_allocations_;
  std::vector<Literal> literals_;
};

// List of all serdes implementations to test.
using Implementations = ::testing::Types<ThunkSequenceSerDesProtobuf>;

TYPED_TEST_SUITE(ThunkSequenceSerdesTest, Implementations);

TYPED_TEST(ThunkSequenceSerdesTest, SerializeAndDeserialize) {
  TF_ASSERT_OK_AND_ASSIGN(ThunkSequence thunk_sequence,
                          this->CreateThunkSequenceFromAllThunkTypes());
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          this->Serialize(thunk_sequence));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ThunkSequence> deserialized,
                          this->Deserialize(serialized));
  EXPECT_TRUE(this->VerifyThunkSequenceEquality(thunk_sequence, *deserialized));
}

}  // namespace

}  // namespace zkx::cpu
