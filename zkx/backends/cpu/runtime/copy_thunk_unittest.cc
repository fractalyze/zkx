/* Copyright 2024 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/backends/cpu/runtime/copy_thunk.h"

#include "gtest/gtest.h"

#include "zkx/backends/cpu/runtime/thunk_testlib.h"
#include "zkx/literal_util.h"
#include "zkx/shape_util.h"

namespace zkx::cpu {

TEST(CopyThunkTest, CopyEmptyShape) {
  auto src = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}});
  auto dst = LiteralUtil::CreateR2<int32_t>({{0, 0}, {0, 0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);
  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);

  BufferAllocation::Slice src_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);
  BufferAllocation::Slice dst_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);

  Shape shape = ShapeUtil::MakeShape(S32, {0, 2});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      CopyThunk::Create({"copy"}, src_slice, shape, dst_slice, shape));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());
}

TEST(CopyThunkTest, CopySameShape) {
  auto src = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}});
  auto dst = LiteralUtil::CreateR2<int32_t>({{0, 0}, {0, 0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);

  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);
  auto [src_slice, dst_slice] =
      CreateBufferAllocationSlice(src_alloc, dst_alloc);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CopyThunk::Create({"copy"}, src_slice, src.shape(), dst_slice,
                                    dst.shape()));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(src, dst);
}

TEST(CopyThunkTest, CopyTransposed) {
  auto src = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}});
  auto dst = LiteralUtil::CreateR2<int32_t>({{0, 0}, {0, 0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);

  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);
  auto [src_slice, dst_slice] =
      CreateBufferAllocationSlice(src_alloc, dst_alloc);

  Shape transposed_shape = src.shape();
  *transposed_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CopyThunk::Create({"copy"}, src_slice, transposed_shape,
                                    dst_slice, dst.shape()));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());

  EXPECT_EQ(dst, LiteralUtil::CreateR2<int32_t>({{1, 3}, {2, 4}}));
}

TEST(CopyThunkTest, CopyTransposedEmptyShape) {
  auto src = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}});
  auto dst = LiteralUtil::CreateR2<int32_t>({{0, 0}, {0, 0}});

  BufferAllocations allocations = CreateBufferAllocations(src, dst);
  auto [src_alloc, dst_alloc] = CreateBufferAllocation(src, dst);

  BufferAllocation::Slice src_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);
  BufferAllocation::Slice dst_slice =
      CreateBufferAllocationSlice(src_alloc, 0, 0);

  Shape shape = ShapeUtil::MakeShape(S32, {0, 2});

  Shape transposed_shape = shape;
  *transposed_shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, CopyThunk::Create({"copy"}, src_slice, transposed_shape,
                                    dst_slice, shape));

  Thunk::ExecuteParams params = {nullptr, &allocations};

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError());
}

}  // namespace zkx::cpu
