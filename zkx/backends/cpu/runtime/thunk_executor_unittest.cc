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

#include "zkx/backends/cpu/runtime/thunk_executor.h"

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/thread_pool.h"
#include "zkx/backends/cpu/runtime/thunk_testlib.h"
#include "zkx/literal_util.h"

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"

namespace zkx::cpu {

using ::testing::ElementsAre;

namespace {

// An adaptor from a lambda that runs tasks and a TaskRunner API.
template <typename Runner, typename WorkerId>
class TaskRunnerAdaptor : public Thunk::TaskRunner {
 public:
  TaskRunnerAdaptor(Runner runner, WorkerId worker_id)
      : runner_(std::move(runner)), worker_id_(std::move(worker_id)) {}

  void operator()(Thunk::Task task) final { runner_(std::move(task)); }

  std::optional<int64_t> current_worker_id() const final {
    return worker_id_();
  }

 private:
  Runner runner_;
  WorkerId worker_id_;
};

template <typename Runner>
auto MakeTaskRunnerFrom(Runner&& runner) {
  auto no_id = []() { return std::nullopt; };
  return TaskRunnerAdaptor<Runner, decltype(no_id)>(
      std::forward<Runner>(runner), no_id);
}

template <typename Runner, typename WorkerId>
auto MakeTaskRunnerFrom(Runner&& runner, WorkerId&& worker_id) {
  return TaskRunnerAdaptor<Runner, WorkerId>(std::forward<Runner>(runner),
                                             std::forward<WorkerId>(worker_id));
}

ThunkExecutor::Options OptionsForTest() {
  return ThunkExecutor::Options{/*execute_sequential_buffer_threshold=*/0,
                                /*execute_sequential_num_thunks_threshold=*/0};
}

}  // namespace

TEST(ThunkExecutorTest, FifoReadyQueueTest) {
  ThunkExecutor::FifoReadyQueue queue({});

  // Check basic queue properties.
  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0);

  queue.Push(1);
  queue.Push(2);
  queue.Push(3);

  ASSERT_EQ(queue.Size(), 3);

  EXPECT_EQ(queue.Pop(), 1);
  EXPECT_EQ(queue.Pop(), 2);
  EXPECT_EQ(queue.Pop(), 3);

  EXPECT_TRUE(queue.Empty());
  ASSERT_EQ(queue.Size(), 0);

  // Prepare queue for PopHalf test case.
  queue.Push(1);
  queue.Push(2);
  queue.Push(3);

  // Pop half of the queue.
  ThunkExecutor::FifoReadyQueue half0 = queue.PopHalf();
  ASSERT_EQ(half0.Size(), 2);
  EXPECT_EQ(half0.Pop(), 2);
  EXPECT_EQ(half0.Pop(), 3);

  // Check that the rest is still in the queue.
  ASSERT_EQ(queue.Size(), 1);

  // Pop the rest of the queue.
  ThunkExecutor::FifoReadyQueue half1 = queue.PopHalf();
  ASSERT_EQ(half1.Size(), 1);

  // Check that all nodes were returned from PopHalf.
  EXPECT_EQ(queue.Size(), 0);

  // Add 5 elements to test Pop followed by PopHalf.
  queue.Push(1);
  queue.Push(2);
  queue.Push(3);
  queue.Push(4);
  queue.Push(5);

  EXPECT_EQ(queue.Pop(), 1);

  // Check that PopHalf returns 2 last nodes.
  ThunkExecutor::FifoReadyQueue half2 = queue.PopHalf();
  ASSERT_EQ(half2.Size(), 2);
  EXPECT_EQ(half2.Pop(), 4);
  EXPECT_EQ(half2.Pop(), 5);
}

TEST(ThunkExecutorTest, LifoReadyQueueTest) {
  ThunkExecutor::LifoReadyQueue queue({});

  // Check basic queue properties.
  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0);

  queue.Push(1);
  queue.Push(2);
  queue.Push(3);

  ASSERT_EQ(queue.Size(), 3);

  EXPECT_EQ(queue.Pop(), 3);
  EXPECT_EQ(queue.Pop(), 2);
  EXPECT_EQ(queue.Pop(), 1);

  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0);

  // Prepare queue for PopHalf test case.
  queue.Push(1);
  queue.Push(2);
  queue.Push(3);

  // Pop half of the queue.
  ThunkExecutor::LifoReadyQueue half0 = queue.PopHalf();
  ASSERT_EQ(half0.Size(), 2);
  EXPECT_EQ(half0.Pop(), 2);
  EXPECT_EQ(half0.Pop(), 1);

  // Check that the rest is still in the queue.
  ASSERT_EQ(queue.Size(), 1);

  // Pop the rest of the queue.
  ThunkExecutor::LifoReadyQueue half1 = queue.PopHalf();
  ASSERT_EQ(half1.Size(), 1);

  // ASSERT_EQ that all nodes were returned from PopHalf.
  EXPECT_EQ(queue.Size(), 0);

  // Add 5 elements to test Pop followed by PopHalf.
  queue.Push(1);
  queue.Push(2);
  queue.Push(3);
  queue.Push(4);
  queue.Push(5);

  EXPECT_EQ(queue.Pop(), 5);

  // Check that PopHalf returns first 2 nodes.
  ThunkExecutor::LifoReadyQueue half2 = queue.PopHalf();
  ASSERT_EQ(half2.Size(), 3);
  EXPECT_EQ(half2.Pop(), 3);
  EXPECT_EQ(half2.Pop(), 2);
  EXPECT_EQ(half2.Pop(), 1);
}

TEST(ThunkExecutorTest, PriorityReadyQueueTest) {
  std::vector<ThunkExecutor::NodeDef> nodes_defs(16);
  for (size_t i = 0; i < nodes_defs.size(); ++i) {
    nodes_defs[i].priority = i;
  }

  ThunkExecutor::PriorityReadyQueue queue(nodes_defs, {});
  // Check basic queue properties.
  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0);

  queue.Push(1);
  queue.Push(3);
  queue.Push(2);

  EXPECT_EQ(queue.Pop(), 3);
  EXPECT_EQ(queue.Pop(), 2);
  EXPECT_EQ(queue.Pop(), 1);

  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0);

  // Prepare queue for PopHalf test case.
  queue.Push(2);
  queue.Push(1);
  queue.Push(3);

  // Pop half of the queue.
  ThunkExecutor::PriorityReadyQueue half0 = queue.PopHalf();
  ASSERT_EQ(half0.Size(), 2);
  EXPECT_EQ(half0.Pop(), 2);
  EXPECT_EQ(half0.Pop(), 1);

  // Check that the rest is still in the queue.
  ASSERT_EQ(queue.Size(), 1);

  // Pop the rest of the queue.
  ThunkExecutor::PriorityReadyQueue half1 = queue.PopHalf();
  ASSERT_EQ(half1.Size(), 1);
  EXPECT_EQ(half1.Pop(), 3);

  // Check that all nodes were returned from PopHalf.
  ASSERT_EQ(queue.Size(), 0);

  // Add 5 elements to test Pop followed by PopHalf.
  queue.Push(4);
  queue.Push(3);
  queue.Push(5);
  queue.Push(1);
  queue.Push(2);

  EXPECT_EQ(queue.Pop(), 5);

  // Check that PopHalf returns 2 last nodes.
  ThunkExecutor::PriorityReadyQueue half2 = queue.PopHalf();
  ASSERT_EQ(half2.Size(), 2);
  EXPECT_EQ(half2.Pop(), 2);
  EXPECT_EQ(half2.Pop(), 1);
}

TEST(ThunkExecutorTest, DependencyOrdering) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/40);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/40, /*size=*/40);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/20, /*size=*/40);

  ThunkSequence sequence;
  sequence.push_back(AddI32Thunk::Create("a", {slice0}, {slice0}));
  sequence.push_back(AddI32Thunk::Create("b", {slice1}, {slice1}));
  sequence.push_back(AddI32Thunk::Create("c", {slice2}, {slice2}));

  TF_ASSERT_OK_AND_ASSIGN(
      ThunkExecutor executor,
      ThunkExecutor::Create(std::move(sequence), OptionsForTest()));

  EXPECT_FALSE(executor.is_sequential());
  EXPECT_THAT(executor.source(), ElementsAre(0, 1));
  EXPECT_THAT(executor.sink(), ElementsAre(2));

  EXPECT_EQ(executor.node_def(0).priority, 1);
  EXPECT_EQ(executor.node_def(1).priority, 1);
  EXPECT_EQ(executor.node_def(2).priority, 0);
}

TEST(ThunkExecutorTest, SequentialOrdering) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/40);

  ThunkSequence sequence;
  sequence.push_back(AddI32Thunk::Create("a", {slice}, {slice}));
  sequence.push_back(AddI32Thunk::Create("b", {slice}, {slice}));
  sequence.push_back(AddI32Thunk::Create("c", {slice}, {slice}));

  TF_ASSERT_OK_AND_ASSIGN(
      ThunkExecutor executor,
      ThunkExecutor::Create(std::move(sequence), OptionsForTest()));

  EXPECT_TRUE(executor.is_sequential());
  EXPECT_THAT(executor.source(), ElementsAre(0));
  EXPECT_THAT(executor.sink(), ElementsAre(2));

  EXPECT_EQ(executor.node_def(0).priority, 2);
  EXPECT_EQ(executor.node_def(1).priority, 1);
  EXPECT_EQ(executor.node_def(2).priority, 0);
}

TEST(ThunkExecutorTest, ResourceOrdering) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/40);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/40, /*size=*/40);

  ThunkSequence sequence;
  sequence.push_back(AddI32Thunk::Create("a", {slice0}, {slice0},
                                         /*trace=*/nullptr,
                                         /*use_shared_resource=*/true));
  sequence.push_back(AddI32Thunk::Create("b", {slice1}, {slice1},
                                         /*trace=*/nullptr,
                                         /*use_shared_resource=*/true));

  TF_ASSERT_OK_AND_ASSIGN(
      ThunkExecutor executor,
      ThunkExecutor::Create(std::move(sequence), OptionsForTest()));

  EXPECT_TRUE(executor.is_sequential());
  EXPECT_THAT(executor.source(), ElementsAre(0));
  EXPECT_THAT(executor.sink(), ElementsAre(1));

  EXPECT_EQ(executor.node_def(0).priority, 1);
  EXPECT_EQ(executor.node_def(1).priority, 0);
}

TEST(ThunkExecutorTest, TransitiveReduction) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/40);

  ThunkSequence sequence;
  sequence.push_back(AddI32Thunk::Create("a", {slice}, {slice}));
  sequence.push_back(AddI32Thunk::Create("b", {slice}, {slice}));
  sequence.push_back(AddI32Thunk::Create("c", {slice}, {slice}));

  TF_ASSERT_OK_AND_ASSIGN(
      ThunkExecutor executor,
      ThunkExecutor::Create(std::move(sequence), OptionsForTest()));

  EXPECT_THAT(executor.source(), ElementsAre(0));
  EXPECT_THAT(executor.sink(), ElementsAre(2));

  EXPECT_THAT(executor.node_def(0).out_edges, ElementsAre(1));
  EXPECT_THAT(executor.node_def(1).in_edges, ElementsAre(0));
  EXPECT_THAT(executor.node_def(1).out_edges, ElementsAre(2));
  EXPECT_THAT(executor.node_def(2).in_edges, ElementsAre(1));

  EXPECT_EQ(executor.node_def(0).priority, 2);
  EXPECT_EQ(executor.node_def(1).priority, 1);
  EXPECT_EQ(executor.node_def(2).priority, 0);
}

TEST(ThunkExecutorTest, Execute) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/40);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/40, /*size=*/40);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/20, /*size=*/40);

  std::vector<std::string> trace;

  ThunkSequence sequence;
  sequence.push_back(AddI32Thunk::Create("a", {slice0}, {slice0}, &trace));
  sequence.push_back(AddI32Thunk::Create("b", {slice1}, {slice1}, &trace));
  sequence.push_back(AddI32Thunk::Create("c", {slice2}, {slice2}, &trace));

  TF_ASSERT_OK_AND_ASSIGN(
      ThunkExecutor executor,
      ThunkExecutor::Create(std::move(sequence), OptionsForTest()));

  // Shared src and dst allocation.
  auto data = LiteralUtil::CreateFull({20}, int32_t{1});
  BufferAllocations allocations = CreateBufferAllocations(data);

  auto task_runner = MakeTaskRunnerFrom(
      [&](Thunk::Task task) {
        trace.push_back("<TaskRunner>");
        task();
      },
      // Always return current worker id as 0.
      [] { return 0; });

  Thunk::ExecuteParams params = {nullptr, &allocations};
  params.task_runner = &task_runner;
  params.session =
      Thunk::ExecuteSession(/*max_workers=*/8, /*split_threshold=*/0);

  auto execute_event = executor.Execute(params);

  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsConcrete());

  EXPECT_THAT(trace, ElementsAre("<TaskRunner>", "b", "a", "c"));
  EXPECT_EQ(data, LiteralUtil::CreateR1<int32_t>({2, 2, 2, 2, 2,     // slice0
                                                  4, 4, 4, 4, 4,     // slice2
                                                  4, 4, 4, 4, 4,     // ...
                                                  2, 2, 2, 2, 2}));  // slice1
}

namespace {

//===----------------------------------------------------------------------===//
// ThunkExecutor resource isolation testing
//===----------------------------------------------------------------------===//

// No-op thunk that completes execution on a separate thread pool. We use this
// thunk to test that ThunkExecutor can jump out of a separate thread pool to
// continue execution in the intra-op thread pool. This is important for
// resource isolation as we don't want to accidentally continue with expensive
// execution on a non blocking IO callbacks thread pool.
class NoOpAsyncThunk : public Thunk {
 public:
  NoOpAsyncThunk(std::string name, BufferAllocation::Slice slice)
      : Thunk(Kind::kKernel, Info{std::move(name)}), slice_(slice) {}

  static std::unique_ptr<NoOpAsyncThunk> Create(std::string name,
                                                BufferAllocation::Slice slice) {
    return std::make_unique<NoOpAsyncThunk>(std::move(name), slice);
  }

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams&) final {
    auto ret = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();
    ThreadPool()->Schedule([ret] {
      tsl::Env::Default()->Sleep(absl::Microseconds(10 * 1000));
      ret.SetStateConcrete();
    });
    return ret;
  }

  BufferUses buffer_uses() const override {
    return BufferUses{BufferUse::Write(slice_)};
  }

 private:
  static tsl::thread::ThreadPool* ThreadPool() {
    static auto* thread_pool =
        new tsl::thread::ThreadPool(tsl::Env::Default(), "no-op-thunk", 8);
    return thread_pool;
  }

  BufferAllocation::Slice slice_;
};

}  // namespace

TEST(ThunkExecutorTest, ExecuteOnCorrectThreadPool) {
  BufferAllocation alloc(/*index=*/0, /*size=*/60, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/20);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/20, /*size=*/20);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/40, /*size=*/20);

  std::array<BufferAllocation::Slice, 3> slices = {slice0, slice1, slice2};

  ThunkSequence sequence;
  for (int i = 0; i < 100; ++i) {
    sequence.push_back(NoOpAsyncThunk::Create(absl::StrCat(i), slices[i % 3]));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      ThunkExecutor executor,
      ThunkExecutor::Create(std::move(sequence), OptionsForTest()));

  auto data = LiteralUtil::CreateFull({60}, uint8_t{1});
  BufferAllocations allocations = CreateBufferAllocations(data);

  // Task runner must be used only when ThunkExecutor detects that it runs on a
  // wrong thread and has to jump into the task runner.
  std::atomic<int32_t> num_tasks = 0;
  auto task_runner = MakeTaskRunnerFrom([&](Thunk::Task task) {
    ++num_tasks;
    task();
  });

  Thunk::ExecuteParams params = {nullptr, &allocations};
  params.task_runner = &task_runner;
  params.session =
      Thunk::ExecuteSession(/*max_workers=*/1, /*split_threshold=*/1000);

  auto execute_event = executor.Execute(params);

  tsl::BlockUntilReady(execute_event);
  ASSERT_TRUE(execute_event.IsConcrete());

  // We compare using GE because thread scheduling introduces small
  // non-determinism and ThunkExecutor might resume after NoOpAsyncThunk already
  // completes its execution event.
  EXPECT_GE(num_tasks, 90);
}

}  // namespace zkx::cpu
