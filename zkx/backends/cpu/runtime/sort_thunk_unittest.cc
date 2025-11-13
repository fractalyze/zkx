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

#include "zkx/backends/cpu/runtime/sort_thunk.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"

#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/backends/cpu/runtime/buffer_allocations.h"
#include "zkx/backends/cpu/runtime/function_library.h"
#include "zkx/backends/cpu/runtime/thunk_testlib.h"
#include "zkx/base/random.h"
#include "zkx/layout.h"
#include "zkx/layout_util.h"
#include "zkx/literal.h"
#include "zkx/literal_util.h"
#include "zkx/service/buffer_assignment.h"
#include "zkx/shape.h"

namespace zkx::cpu {
namespace {

class SortThunkTest : public testing::TestWithParam<bool> {};

// Sorts the data using only the first input (that uint32_t!).
bool LessThan(const void** data) {
  auto* lhs = reinterpret_cast<const uint32_t*>(data[0]);
  auto* rhs = reinterpret_cast<const uint32_t*>(data[1]);
  return *lhs < *rhs;
}

class LessThanComparator : public FunctionLibrary {
 public:
  absl::StatusOr<void*> ResolveFunction(TypeId type_id,
                                        std::string_view name) final {
    DCHECK_EQ(name, "less_than");
    return reinterpret_cast<void*>(LessThanWrapper);
  }

 private:
  static bool LessThanWrapper(const void** data) { return LessThan(data); }
};

TEST_P(SortThunkTest, DescendingSortPlainArray) {
  bool is_stable = GetParam();

  Literal data = LiteralUtil::CreateR1<uint32_t>(
      base::CreateVector(10000, []() { return base::Uniform<uint32_t>(); }));

  BufferAllocations allocations = CreateBufferAllocations(data);
  BufferAllocation alloc = CreateBufferAllocation(0, data);
  BufferAllocation::Slice slice = CreateBufferAllocationSlice(alloc);

  // The comparator function is not used in the plain array sort when the sort
  // direction is specified and data types are supported.
  auto fake_less_than = [](const void** data) { return false; };

  // Use sort direction to activate the most efficient sorting function.
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, SortThunk::Create({"sort"}, {{slice, data.shape()}},
                                    /*dimension=*/0, is_stable, fake_less_than,
                                    SortThunk::SortDirection::kDescending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_TRUE(std::is_sorted(data.data<uint32_t>().begin(),
                             data.data<uint32_t>().end(),
                             std::greater<uint32_t>()));
}

TEST_P(SortThunkTest, Sort1D) {
  bool is_stable = GetParam();

  auto data = LiteralUtil::CreateR1<uint32_t>({2, 4, 1, 3});
  auto indices = LiteralUtil::CreateR1<int32_t>({0, 1, 2, 3});

  BufferAllocations allocations = CreateBufferAllocations(data, indices);

  auto [alloc0, alloc1] = CreateBufferAllocation(data, indices);
  auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      SortThunk::Create({"sort"},
                        {{slice0, data.shape()}, {slice1, indices.shape()}},
                        /*dimension=*/0, is_stable, LessThan,
                        SortThunk::SortDirection::kAscending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(data, LiteralUtil::CreateR1<uint32_t>({1, 2, 3, 4}));
  EXPECT_EQ(indices, LiteralUtil::CreateR1<int32_t>({2, 0, 3, 1}));
}

TEST_P(SortThunkTest, Sort1DDynamicNumInputs) {
  bool is_stable = GetParam();

  Literal data = LiteralUtil::CreateR1<uint32_t>(
      {17, 16, 5,  10, 30, 8,  9,  21, 14, 32, 29, 28, 19, 12, 25, 22,
       18, 35, 34, 23, 7,  13, 26, 33, 15, 24, 20, 31, 6,  27, 11});

  Literal indices = LiteralUtil::CreateR1<int32_t>(
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
       16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});

  // We use dummy data to create large number of input to trigger the dynamic
  // sort implementation, but we don't use it for sorting.
  Literal dummy_data = LiteralUtil::CreateR1<uint32_t>(base::CreateVector(
      data.element_count(), []() { return base::Uniform<uint32_t>(); }));

  BufferAllocations allocations =
      CreateBufferAllocations(data, indices, dummy_data);

  auto [data_alloc, indices_alloc, dummy_alloc] =
      CreateBufferAllocation(data, indices, dummy_data);
  auto [data_slice, indices_slice, dummy_slice] =
      CreateBufferAllocationSlice(data_alloc, indices_alloc, dummy_alloc);

  // We use only first input for sorting, the rest of the inputs are shuffled
  // according to the values in the `data` literal.
  std::vector<SortThunk::Input> inputs = {{data_slice, data.shape()},
                                          {indices_slice, indices.shape()}};
  inputs.resize(40, {dummy_slice, dummy_data.shape()});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, SortThunk::Create({"sort"}, inputs,
                                    /*dimension=*/0, is_stable, LessThan,
                                    SortThunk::SortDirection::kAscending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  auto expected_data = LiteralUtil::CreateR1<uint32_t>(
      {5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35});

  auto expected_indices = LiteralUtil::CreateR1<int32_t>(
      {2, 28, 20, 5,  6,  3,  30, 13, 21, 8, 24, 1, 0,  16, 12, 26,
       7, 15, 19, 25, 14, 22, 29, 11, 10, 4, 27, 9, 23, 18, 17});

  EXPECT_EQ(data, expected_data);
  EXPECT_EQ(indices, expected_indices);
}

namespace {

void Sort2DHelper(bool is_stable, int64_t dimension, Literal& data,
                  Literal& indices, const Literal& expected_data,
                  const Literal& expected_indices) {
  BufferAllocations allocations = CreateBufferAllocations(data, indices);

  auto [alloc0, alloc1] = CreateBufferAllocation(data, indices);
  auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

  TF_ASSERT_OK_AND_ASSIGN(
      auto sort,
      SortThunk::Create({"sort"},
                        {{slice0, data.shape()}, {slice1, indices.shape()}},
                        dimension, is_stable, "less_than",
                        SortThunk::SortDirection::kAscending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  LessThanComparator less_than_comparator;
  params.function_library = &less_than_comparator;

  auto execute_event = sort->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(data, expected_data);
  EXPECT_EQ(indices, expected_indices);
}

}  // namespace

TEST_P(SortThunkTest, Sort2D) {
  bool is_stable = GetParam();

  auto data = LiteralUtil::CreateR2<uint32_t>({{2, 4}, {1, 3}});
  auto indices = LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 3}});
  Sort2DHelper(is_stable, 0, data, indices,
               LiteralUtil::CreateR2<uint32_t>({{1, 3}, {2, 4}}),
               LiteralUtil::CreateR2<int32_t>({{2, 3}, {0, 1}}));

  data = LiteralUtil::CreateR2<uint32_t>({{4, 3}, {2, 1}});
  indices = LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 3}});
  Sort2DHelper(is_stable, 1, data, indices,
               LiteralUtil::CreateR2<uint32_t>({{3, 4}, {1, 2}}),
               LiteralUtil::CreateR2<int32_t>({{1, 0}, {3, 2}}));
}

namespace {

void Sort2DWithLayoutHelper(bool is_stable, int64_t dimension, Literal& data,
                            Literal& indices, const Literal& expected_data,
                            const Literal& expected_indices) {
  BufferAllocations allocations = CreateBufferAllocations(data, indices);

  auto [alloc0, alloc1] = CreateBufferAllocation(data, indices);
  auto [slice0, slice1] = CreateBufferAllocationSlice(alloc0, alloc1);

  Shape data_shape = data.shape();
  *data_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  Shape indices_shape = indices.shape();
  *indices_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  TF_ASSERT_OK_AND_ASSIGN(
      auto sort,
      SortThunk::Create(
          {"sort"}, {{slice0, data_shape}, {slice1, indices_shape}}, dimension,
          is_stable, "less_than", SortThunk::SortDirection::kAscending));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;

  LessThanComparator less_than_comparator;
  params.function_library = &less_than_comparator;

  auto execute_event = sort->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(data, expected_data);
  EXPECT_EQ(indices, expected_indices);
}

}  // namespace

TEST_P(SortThunkTest, Sort2DWithLayout) {
  bool is_stable = GetParam();

  auto data = LiteralUtil::CreateR2<uint32_t>({{4, 3}, {2, 1}});
  auto indices = LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 3}});

  Sort2DWithLayoutHelper(is_stable, 0, data, indices,
                         LiteralUtil::CreateR2<uint32_t>({{3, 4}, {1, 2}}),
                         LiteralUtil::CreateR2<int32_t>({{1, 0}, {3, 2}}));

  data = LiteralUtil::CreateR2<uint32_t>({{2, 4}, {1, 3}});
  indices = LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 3}});

  Sort2DWithLayoutHelper(is_stable, 1, data, indices,
                         LiteralUtil::CreateR2<uint32_t>({{1, 3}, {2, 4}}),
                         LiteralUtil::CreateR2<int32_t>({{2, 3}, {0, 1}}));
}

INSTANTIATE_TEST_SUITE_P(SortThunk, SortThunkTest, testing::Bool(),
                         testing::PrintToStringParamName());

}  // namespace
}  // namespace zkx::cpu
