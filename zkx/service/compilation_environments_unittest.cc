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

#include "zkx/service/compilation_environments.h"

#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/service/test_compilation_environment.pb.h"

namespace zkx::test {
namespace {

// In order to use TestCompilationEnvironment* with CompilationEnvironments, we
// must define ProcessNewEnv for them.
std::unique_ptr<google::protobuf::Message> ProcessNewEnv1(
    std::unique_ptr<google::protobuf::Message> msg) {
  std::unique_ptr<TestCompilationEnvironment1> env(
      tensorflow::down_cast<TestCompilationEnvironment1*>(msg.release()));
  if (!env) {
    env = std::make_unique<TestCompilationEnvironment1>();
  }
  if (env->some_flag() == 0 || env->some_flag() == 1) {
    env->set_some_flag(100);
  }
  return env;
}
std::unique_ptr<google::protobuf::Message> ProcessNewEnv2(
    std::unique_ptr<google::protobuf::Message> msg) {
  std::unique_ptr<TestCompilationEnvironment2> env(
      tensorflow::down_cast<TestCompilationEnvironment2*>(msg.release()));
  if (!env) {
    env = std::make_unique<TestCompilationEnvironment2>();
  }
  if (env->some_other_flag() == 0) {
    env->set_some_other_flag(200);
  }
  return env;
}
std::unique_ptr<google::protobuf::Message> ProcessNewEnv3(
    std::unique_ptr<google::protobuf::Message> msg) {
  std::unique_ptr<TestCompilationEnvironment3> env(
      tensorflow::down_cast<TestCompilationEnvironment3*>(msg.release()));
  if (!env) {
    env = std::make_unique<TestCompilationEnvironment3>();
  }
  if (env->a_third_flag() == 0) {
    env->set_a_third_flag(300);
  }
  return env;
}

class CompilationEnvironmentsTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    CompilationEnvironments::RegisterProcessNewEnvFn(
        TestCompilationEnvironment1::descriptor(), ProcessNewEnv1);
    CompilationEnvironments::RegisterProcessNewEnvFn(
        TestCompilationEnvironment2::descriptor(), ProcessNewEnv2);
    CompilationEnvironments::RegisterProcessNewEnvFn(
        TestCompilationEnvironment3::descriptor(), ProcessNewEnv3);
  }
};

}  // namespace

TEST_F(CompilationEnvironmentsTest, GetDefaultEnv) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, GetDefaultMutableEnv) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, GetAddedEnvNotModifiedByProcessNewEnv) {
  CompilationEnvironments envs;
  auto env = std::make_unique<TestCompilationEnvironment1>();
  env->set_some_flag(5);
  ASSERT_TRUE(envs.AddEnv(std::move(env)).ok());
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 5);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 5);
}

TEST_F(CompilationEnvironmentsTest, GetAddedEnvModifiedByProcessNewEnv) {
  CompilationEnvironments envs;
  auto env = std::make_unique<TestCompilationEnvironment1>();
  env->set_some_flag(1);
  ASSERT_TRUE(envs.AddEnv(std::move(env)).ok());
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, MultipleEnvs) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment2>().some_other_flag(), 200);
  EXPECT_EQ(envs.GetEnv<TestCompilationEnvironment1>().some_flag(), 100);
}

TEST_F(CompilationEnvironmentsTest, MultipleMutableEnvs) {
  CompilationEnvironments envs;
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 100);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment2>().some_other_flag(),
            200);
  envs.GetMutableEnv<TestCompilationEnvironment1>().set_some_flag(101);
  envs.GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(201);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment1>().some_flag(), 101);
  EXPECT_EQ(envs.GetMutableEnv<TestCompilationEnvironment2>().some_other_flag(),
            201);
}

TEST_F(CompilationEnvironmentsTest, CopyConstructor) {
  // Setup envs with 2 environments
  auto envs = std::make_unique<CompilationEnvironments>();
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(10);
  ASSERT_TRUE(envs->AddEnv(std::move(env1)).ok());
  auto env2 = std::make_unique<TestCompilationEnvironment2>();
  ASSERT_TRUE(envs->AddEnv(std::move(env2)).ok());
  envs->GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(20);

  // Call the copy constructor and delete the original CompilationEnvironments
  auto envs_copy = std::make_unique<CompilationEnvironments>(*envs);
  envs.reset();

  // Verify that envs_copy has the same values with which envs was initialized
  EXPECT_EQ(envs_copy->GetEnv<TestCompilationEnvironment1>().some_flag(), 10);
  EXPECT_EQ(envs_copy->GetEnv<TestCompilationEnvironment2>().some_other_flag(),
            20);
}

TEST_F(CompilationEnvironmentsTest, CopyAssignment) {
  // Setup envs1 with 2 environments
  auto envs1 = std::make_unique<CompilationEnvironments>();
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(10);
  ASSERT_TRUE(envs1->AddEnv(std::move(env1)).ok());
  auto env2 = std::make_unique<TestCompilationEnvironment2>();
  ASSERT_TRUE(envs1->AddEnv(std::move(env2)).ok());
  envs1->GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(20);

  // Create envs2 with some environments that should be deleted on copy
  // assignment
  auto envs2 = std::make_unique<CompilationEnvironments>();
  auto env3 = std::make_unique<TestCompilationEnvironment1>();
  env3->set_some_flag(30);
  ASSERT_TRUE(envs2->AddEnv(std::move(env3)).ok());
  auto env4 = std::make_unique<TestCompilationEnvironment3>();
  env4->set_a_third_flag(40);
  ASSERT_TRUE(envs2->AddEnv(std::move(env4)).ok());

  // Assign envs1 to envs2, and delete envs1. After assignment, the environments
  // originally added to envs2 should be deleted, and copies of the environments
  // in envs1 should be added to envs2.
  *envs2 = *envs1;
  envs1.reset();

  // Verify that envs2 has the same values with which envs1 was initialized
  EXPECT_EQ(envs2->GetEnv<TestCompilationEnvironment1>().some_flag(), 10);
  EXPECT_EQ(envs2->GetEnv<TestCompilationEnvironment2>().some_other_flag(), 20);

  // Since envs1 did not have TestCompilationEnvironment3, after copy
  // assignment, envs2 will not have one either. So, we should get the default
  // environment value.
  EXPECT_EQ(envs2->GetEnv<TestCompilationEnvironment3>().a_third_flag(), 300);
}

TEST_F(CompilationEnvironmentsTest, ProtoRoundTrip) {
  // Setup envs with 2 environments.
  auto envs = std::make_unique<CompilationEnvironments>();
  auto env1 = std::make_unique<TestCompilationEnvironment1>();
  env1->set_some_flag(10);
  ASSERT_TRUE(envs->AddEnv(std::move(env1)).ok());
  auto env2 = std::make_unique<TestCompilationEnvironment2>();
  ASSERT_TRUE(envs->AddEnv(std::move(env2)).ok());
  envs->GetMutableEnv<TestCompilationEnvironment2>().set_some_other_flag(20);

  auto proto = envs->ToProto();
  TF_ASSERT_OK_AND_ASSIGN(auto envs_deserialized,
                          CompilationEnvironments::CreateFromProto(proto));

  // Verify that envs_deserialized has the same values with which envs was
  // initialized.
  EXPECT_EQ(
      envs_deserialized->GetEnv<TestCompilationEnvironment1>().some_flag(), 10);
  EXPECT_EQ(envs_deserialized->GetEnv<TestCompilationEnvironment2>()
                .some_other_flag(),
            20);
}

TEST_F(CompilationEnvironmentsTest, EnvTypePresenceCheck) {
  CompilationEnvironments envs;
  EXPECT_FALSE(envs.HasEnv<TestCompilationEnvironment1>());
  envs.GetEnv<TestCompilationEnvironment1>();
  EXPECT_TRUE(envs.HasEnv<TestCompilationEnvironment1>());
}

}  // namespace zkx::test
