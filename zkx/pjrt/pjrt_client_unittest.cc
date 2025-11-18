/* Copyright 2022 The OpenXLA Authors.
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

#include "zkx/pjrt/pjrt_client.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "zkx/cpu_function_runtime.h"
#include "zkx/literal_util.h"
#include "zkx/pjrt/pjrt_test_client.h"
#include "zkx/tests/literal_test_util.h"

namespace zkx {
namespace {

// TODO(chokobole): Uncomment this. Dependency: ZkxBuilder, GetTupleElement
// std::unique_ptr<PjRtLoadedExecutable> MakeIncrementProgram(
//     PjRtClient* client, bool alias, int device, bool tuplize_arg = false) {
//   Shape shape = ShapeUtil::MakeShape(S32, {4});
//   ZkxBuilder builder("inc");
//   if (tuplize_arg) {
//     shape = ShapeUtil::MakeTupleShape({shape});
//   }
//   auto inp = Parameter(&builder, 0, shape, "inp");
//   if (tuplize_arg) {
//     inp = GetTupleElement(inp, 0);
//   }
//   auto one = ConstantR0<int32_t>(&builder, 1);
//   auto inc = Add(inp, one);
//   if (alias) {
//     builder.SetUpAlias({}, 0, {});
//   }
//   ZkxComputation computation = builder.Build(inc).value();
//   DeviceAssignment assignment(1, 1);
//   assignment(0, 0) = device;
//   CompileOptions options;
//   options.parameter_is_tupled_arguments = tuplize_arg;
//   options.executable_build_options.set_device_assignment(assignment);
//   return client->Compile(computation, options).value();
// }

class PjRtClientTest
    : public ::testing::TestWithParam<ExecuteOptions::ExecutionMode> {};

}  // namespace

// TODO(chokobole): Add test. Dependency: MakeIncrementProgram
// TEST_P(PjRtClientTest, Execute) {

// TODO(chokobole): Add test. Dependency: MakeIncrementProgram
// TEST_P(PjRtClientTest, ExecuteWithImmutableUntilTransferCompletes) {

// TODO(chokobole): Add test. Dependency: MakeIncrementProgram
// TEST_P(PjRtClientTest, ExecuteWithTupleZeroCopy) {

// TODO(chokobole): Add test. Dependency: MakeIncrementProgram
// TEST_P(PjRtClientTest, ExecuteWithDonation) {

// TODO(chokobole): Add test. Dependency: MakeIncrementProgram
// TEST_P(PjRtClientTest, ExecuteWithDonationAbort) {

// TODO(chokobole): Add test. Dependency: MakeIncrementProgram
// TEST_P(PjRtClientTest, ExecuteWithConcurrentUsage) {

// TODO(chokobole): Add test. Dependency: MakeIncrementProgram
// TEST_P(PjRtClientTest, ExecuteWithConcurrentUsageAndDonation)

// TODO(chokobole): Add test. Dependency: MakeIncrementProgram
// INSTANTIATE_TEST_SUITE_P(
//     PjRtClientTestSuite, PjRtClientTest,
//     ::testing::Values(ExecuteOptions::ExecutionMode::kSynchronous,
//                       ExecuteOptions::ExecutionMode::kAsynchronous));

TEST(PjRtClientTest, CopyToDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetPjRtTestClient());
  ASSERT_GT(client->addressable_devices().size(), 1);

  std::vector<int32_t> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  auto* device_1 = client->addressable_devices()[1];

  TF_ASSERT_OK_AND_ASSIGN(auto result, buffer->CopyToMemorySpace(
                                           *device_1->default_memory_space()));

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());

  std::vector<int32_t> expected(4, 0);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST(PjRtClientTest, CopyToDeviceAsync) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetPjRtTestClient());
  ASSERT_GT(client->addressable_devices().size(), 1);

  std::vector<int32_t> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  auto* device_1 = client->addressable_devices()[1];

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "CopyToDeviceAsync",
                                      kNumThreads);

  constexpr int kConcurrentCopy = 16;
  std::vector<std::unique_ptr<PjRtBuffer>> results(kConcurrentCopy);
  for (int i = 0; i < kConcurrentCopy; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(results[i], buffer->CopyToMemorySpace(
                                            *device_1->default_memory_space()));
  }

  // The destructor of TfrtCpuBuffer should wait for outstanding copy.
  buffer.reset();

  for (const auto& result : results) {
    ASSERT_TRUE(result);
    TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());

    std::vector<int32_t> expected(4, 0);
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                       *literal));
  }
}

TEST(PjRtClientTest, CopyToDeviceAsyncExternalCpuOnly) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetPjRtTestClient());
  ASSERT_GT(client->addressable_devices().size(), 1);

  // Skip non-CPU platforms.
  if (client->platform_id() != CpuId()) {
    GTEST_SKIP() << "This test is for CPU only.";
  }

  alignas(cpu_function_runtime::MinAlign()) std::array<int32_t, 4> data;
  data.fill(0);
  auto* data_ptr = data.data();
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->CreateViewOfDeviceBuffer(
          data_ptr, shape, client->memory_spaces()[0],
          /*on_delete_callback=*/[data = std::move(data)]() mutable {
            (void)data;
          }));

  auto* device_1 = client->addressable_devices()[1];

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                      "CopyToDeviceAsyncExternal", kNumThreads);

  constexpr int kConcurrentCopy = 16;
  std::vector<std::unique_ptr<PjRtBuffer>> results(kConcurrentCopy);
  for (int i = 0; i < kConcurrentCopy; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(results[i], buffer->CopyToMemorySpace(
                                            *device_1->default_memory_space()));
  }

  // The destructor of TfrtCpuBuffer should wait for outstanding copy.
  buffer.reset();

  for (const auto& result : results) {
    ASSERT_TRUE(result);
    TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());

    std::vector<int32_t> expected(4, 0);
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                       *literal));
  }
}

TEST(PjRtClientTest, CreateViewOfUnalignedBufferReturnsErrorCpuOnly) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetPjRtTestClient());
  ASSERT_GT(client->addressable_devices().size(), 1);

  // Skip non-CPU platforms.
  if (client->platform_id() != CpuId()) {
    GTEST_SKIP() << "This test is for CPU only.";
  }

  alignas(cpu_function_runtime::MinAlign()) std::array<int32_t, 5> data;
  auto* data_ptr = data.data();

  // Pointer to the second element is always unaligned, because it's shifted by
  // 4 bytes (size of int32_t) from the original pointer.
  auto* unaligned_ptr = data_ptr + 1;

  // Shape with a size smaller than the original data vector, because the
  // 'unaligned_ptr' points to the second element.
  Shape shape = ShapeUtil::MakeShape(S32, {4});

  // Attempt to create a view of the unaligned buffer. Expect an error.
  auto result = client->CreateViewOfDeviceBuffer(
      unaligned_ptr, shape, client->memory_spaces()[0],
      /*on_delete_callback=*/std::function<void()>());

  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr("unaligned data"));
}

// TODO(chokobole): Add test. Dependency: HloModule::ToProto
// TEST(PjRtClientTest, DuplicateDonationError) {

}  // namespace zkx
