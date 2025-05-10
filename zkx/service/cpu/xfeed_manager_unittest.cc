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

#include "zkx/service/cpu/xfeed_manager.h"

#include <stdint.h>

#include <string>

#include "absl/log/check.h"
#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/thread_pool.h"
#include "zkx/service/cpu/cpu_runtime.h"
#include "zkx/shape_util.h"

namespace zkx::cpu::runtime {

class TestInfeedBuffer : public XfeedBuffer {
 public:
  explicit TestInfeedBuffer(int32_t length, bool expect_shape_match = true)
      : shape_(ShapeUtil::MakeShape(U8, {length})),
        done_called_(false),
        length_(length),
        expect_shape_match_(expect_shape_match) {}
  ~TestInfeedBuffer() override { EXPECT_TRUE(done_called_); }

  int32_t length() override { return length_; }
  void* data() override { return nullptr; }
  void Done(absl::StatusOr<Shape> shape) override {
    CHECK(!done_called_);
    done_called_ = true;
    TF_ASSERT_OK(shape.status());
    EXPECT_EQ(expect_shape_match_, ShapeUtil::Equal(shape_, shape.value()))
        << "want " << ShapeUtil::HumanString(shape_) << " "
        << (expect_shape_match_ ? "==" : "!=") << " "
        << ShapeUtil::HumanString(shape.value());
    delete this;
  }

  const Shape& shape() const { return shape_; }

 private:
  Shape shape_;
  bool done_called_;
  int32_t length_;
  bool expect_shape_match_;
};

// Performs the acquire/release sequence on the infeed, as the generated CPU
// code would in the process of executing the infeed operation.
void ProcessNextBuffer(int32_t length) {
  auto shape = ShapeUtil::MakeShape(U8, {length});
  std::string bytes = shape.SerializeAsString();
  void* buffer = __zkx_cpu_runtime_AcquireInfeedBufferForDequeue(
      /*run_options=*/nullptr, length, bytes.data(), bytes.size());
  __zkx_cpu_runtime_ReleaseInfeedBufferAfterDequeue(
      /*run_options=*/nullptr, length, buffer, bytes.data(), bytes.size());
}

// Performs the acquire/release sequence on the outfeed, as the generated CPU
// code would in the process of executing the outfeed operation.
void ProcessNextOutfeedBuffer(int32_t length, const Shape& shape) {
  std::string bytes = shape.SerializeAsString();
  void* buffer = __zkx_cpu_runtime_AcquireOutfeedBufferForPopulation(
      /*run_options=*/nullptr, length, bytes.data(), bytes.size());
  __zkx_cpu_runtime_ReleaseOutfeedBufferAfterPopulation(
      /*run_options=*/nullptr, length, buffer, bytes.data(), bytes.size());
}

TEST(InfeedManagerTest, SingleThreadedSequential) {
  TestInfeedBuffer* a = new TestInfeedBuffer(64);
  TestInfeedBuffer* b = new TestInfeedBuffer(32);

  XfeedManager* xfeed = GetXfeedManager(0);

  xfeed->infeed()->EnqueueBuffersAtomically({a});
  xfeed->infeed()->EnqueueBuffersAtomically({b});
  ProcessNextBuffer(a->length());
  ProcessNextBuffer(b->length());
}

TEST(InfeedManagerTest, SingleThreadedInterleaved) {
  TestInfeedBuffer* a = new TestInfeedBuffer(64);
  TestInfeedBuffer* b = new TestInfeedBuffer(32);

  XfeedManager* xfeed = GetXfeedManager(0);

  xfeed->infeed()->EnqueueBuffersAtomically({a});
  ProcessNextBuffer(a->length());
  xfeed->infeed()->EnqueueBuffersAtomically({b});
  ProcessNextBuffer(b->length());
}

TEST(InfeedManagerTest, MultiThreaded) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test", 2);

  XfeedManager* xfeed = GetXfeedManager(0);

  const int32_t length = 64;

  pool.Schedule([&xfeed]() {
    // Spin for 100 milliseconds
    int64_t start_micros = tsl::Env::Default()->NowMicros();
    while (true) {
      int64_t end_micros = tsl::Env::Default()->NowMicros();
      if ((end_micros - start_micros) >= 100000) {  // 100 ms
        break;
      }
    }
    TestInfeedBuffer* a = new TestInfeedBuffer(length);
    xfeed->infeed()->EnqueueBuffersAtomically({a});
  });

  ProcessNextBuffer(length);
}

TEST(InfeedManagerTest, OutfeedBasic) {
  TestInfeedBuffer* b = new TestInfeedBuffer(32,
                                             /*expect_shape_match=*/true);
  XfeedManager* xfeed = GetXfeedManager(0);
  xfeed->outfeed()->EnqueueBuffersAtomically({b});

  ProcessNextOutfeedBuffer(32, ShapeUtil::MakeShape(U8, {32}));
}

TEST(InfeedManagerTest, OutfeedEmpty) {
  TestInfeedBuffer* b = new TestInfeedBuffer(0, /*expect_shape_match=*/true);
  XfeedManager* xfeed = GetXfeedManager(0);
  xfeed->outfeed()->EnqueueBuffersAtomically({b});

  ProcessNextOutfeedBuffer(0, ShapeUtil::MakeShape(U8, {0}));
}

TEST(InfeedManagerTest, OutfeedWrongShape) {
  TestInfeedBuffer* b = new TestInfeedBuffer(32,
                                             /*expect_shape_match=*/false);
  XfeedManager* xfeed = GetXfeedManager(0);
  xfeed->outfeed()->EnqueueBuffersAtomically({b});

  ProcessNextOutfeedBuffer(32, ShapeUtil::MakeShape(U8, {33}));
}

}  // namespace zkx::cpu::runtime
