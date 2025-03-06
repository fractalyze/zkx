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

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/thread_pool.h"
#include "zkx/backends/cpu/runtime/thread_pool_task_runner.h"
#include "zkx/backends/cpu/runtime/thunk_executor.h"
#include "zkx/backends/cpu/runtime/thunk_testlib.h"
#include "zkx/literal_util.h"

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"

namespace zkx::cpu {

namespace {

// We generate random thunk sequences that may or may not use a shared resource.
enum class SharedResourceUse { kNo, kAll, kRandom };

struct GeneratedThunkSequence {
  explicit GeneratedThunkSequence(int64_t num_elements)
      : src(LiteralUtil::CreateFull({num_elements}, int32_t{1})),
        dst(LiteralUtil::CreateFull({num_elements}, int32_t{0})),
        expected(LiteralUtil::CreateFull({num_elements}, int32_t{0})),
        src_alloc(CreateBufferAllocation(0, src)),
        dst_alloc(CreateBufferAllocation(1, dst)),
        expected_shared_resource_value(0),
        expected_literals({&src, &expected}),
        literals({&src, &dst}) {}

  Literal src;
  Literal dst;
  Literal expected;

  BufferAllocation src_alloc;
  BufferAllocation dst_alloc;

  int32_t expected_shared_resource_value;

  std::vector<Literal*> expected_literals;
  std::vector<Literal*> literals;

  ThunkSequence sequence;
};

absl::StatusOr<std::unique_ptr<GeneratedThunkSequence>> GenerateThunkSequence(
    size_t num_elements, size_t num_thunks,
    SharedResourceUse shared_resource_use, bool inject_errors) {
  auto g = std::make_unique<GeneratedThunkSequence>(num_elements);
  g->sequence.reserve(num_thunks);

  std::minstd_rand0 engine;

  std::uniform_int_distribution<size_t> offset_dist(0, num_elements - 1);
  std::uniform_int_distribution<size_t> size_dist(32, 64);
  std::uniform_int_distribution<size_t> use_resource_dist(0, num_thunks / 10);
  std::uniform_int_distribution<size_t> inject_error_dist(0, num_thunks / 10);

  // Returns a random slice of the allocation.
  auto random_slice = [&](BufferAllocation* alloc) {
    size_t start = offset_dist(engine);
    size_t size = std::min(num_elements - start, size_dist(engine));
    return BufferAllocation::Slice(alloc, start * sizeof(int32_t),
                                   size * sizeof(int32_t));
  };

  for (int i = 0; i < num_thunks; ++i) {
    BufferAllocation::Slice src = random_slice(&g->src_alloc);
    BufferAllocation::Slice dst = random_slice(&g->dst_alloc);

    // Pre-compute expected result while building the thunk sequence.
    BufferAllocations allocations =
        CreateBufferAllocations(absl::MakeSpan(g->expected_literals));
    TF_RETURN_IF_ERROR(AddI32Thunk::Execute(&allocations, src, dst));

    bool use_resource = [&] {
      switch (shared_resource_use) {
        case SharedResourceUse::kNo:
          return false;
        case SharedResourceUse::kAll:
          return true;
        case SharedResourceUse::kRandom:
          return use_resource_dist(engine) == 0;
      }
    }();
    if (use_resource) g->expected_shared_resource_value++;

    bool inject_error = inject_errors && inject_error_dist(engine) == 0;
    g->sequence.push_back(AddI32Thunk::Create(absl::StrCat(i), {src}, {dst},
                                              /*trace=*/nullptr, use_resource,
                                              inject_error));
  }

  return g;
}

// Parameterized thunk executor stress tests that builds a random thunk sequence
// and optionally uses a thread pool to execute thunk executor tasks.
class ThunkExecutorStressTest
    : public testing::TestWithParam<
          std::tuple<int32_t, bool, bool, SharedResourceUse, bool, bool>> {
 public:
  void SetUp() override {
    auto& [num_thunks, use_task_runner, use_device, shared_resource_use,
           inject_errors, use_priority_ready_queue] = GetParam();

    use_task_runner_ = use_task_runner;
    use_device_ = use_device;

    // Both the task runner and the intra-op device share the same underlying
    // thread pool, and we test that they do not deadlock each other and
    // everything works via chaining together asynchronous events. It is a
    // common source of deadlocks to wait for the completion of tasks scheduled
    // into the same thread pool where awaiting thread is executing.
    if (use_task_runner_ || use_device_) {
      thread_pool_.emplace(tsl::Env::Default(), "thunk-executor", 8);
      device_.emplace(thread_pool_->AsEigenThreadPool(),
                      thread_pool_->NumThreads());
      task_runner_.emplace(thread_pool_->AsEigenThreadPool());
    }
  }

  Thunk::TaskRunner* task_runner() {
    if (!use_task_runner_) return nullptr;
    return &*task_runner_;
  }

  Eigen::ThreadPoolDevice* device() {
    if (!use_device_) return nullptr;
    return &*device_;
  }

 private:
  bool use_task_runner_;
  bool use_device_;
  std::optional<tsl::thread::ThreadPool> thread_pool_;
  std::optional<Eigen::ThreadPoolDevice> device_;
  std::optional<ThreadPoolTaskRunner> task_runner_;
};

}  // namespace

TEST_P(ThunkExecutorStressTest, Execute) {
  auto [num_thunks, use_task_runner, use_device, shared_resource_use,
        inject_errors, use_priority_ready_queue] = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GeneratedThunkSequence> g,
      GenerateThunkSequence(/*num_elements=*/1024, num_thunks,
                            shared_resource_use, inject_errors));

  ThunkExecutor::Options executor_options = {
      /*execute_sequential_buffer_threshold=*/0,
      /*use_priority_ready_queue=*/use_priority_ready_queue,
  };

  TF_ASSERT_OK_AND_ASSIGN(
      ThunkExecutor executor,
      ThunkExecutor::Create(std::move(g->sequence), executor_options));

  BufferAllocations allocations =
      CreateBufferAllocations(absl::MakeSpan(g->literals));
  // TODO(chokobole): Uncomment this. Dependency: FunctionLibrary, XfeedManager
  // Thunk::ExecuteParams params = {nullptr, &allocations, nullptr, device(),
  //                                task_runner()};
  Thunk::ExecuteParams params = {&allocations, device(), task_runner()};

  InitSharedResource();

  auto execute_event = executor.Execute(params);
  tsl::BlockUntilReady(execute_event);

  if (inject_errors) {
    ASSERT_TRUE(execute_event.IsError());
    EXPECT_EQ(execute_event.GetError(), absl::InternalError("Injected error"));
  } else {
    ASSERT_TRUE(execute_event.IsConcrete());
    EXPECT_EQ(GetSharedResource(), g->expected_shared_resource_value);
    EXPECT_EQ(g->dst, g->expected);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ThunkExecutor, ThunkExecutorStressTest,
    testing::Combine(/*num_thunks=*/testing::ValuesIn({10, 100, 1000}),
                     /*use_task_runner=*/testing::Bool(),
                     /*use_device=*/testing::Bool(),
                     /*shared_resource_use=*/
                     testing::Values(SharedResourceUse::kNo,
                                     SharedResourceUse::kAll,
                                     SharedResourceUse::kRandom),
                     /*inject_errors=*/testing::Bool(),
                     /*use_priority_ready_queue=*/testing::Bool()));

}  // namespace zkx::cpu
