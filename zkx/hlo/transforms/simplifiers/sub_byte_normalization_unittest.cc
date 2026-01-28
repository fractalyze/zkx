/* Copyright 2026 The ZKX Authors.

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

#include "zkx/hlo/transforms/simplifiers/sub_byte_normalization.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "zkx/primitive_util.h"

namespace zkx {
namespace {

class SubByteNormalizationTest : public HloHardwareIndependentTestBase {
 protected:
  void RunSetElementSize(HloModule* module, bool change_expected) {
    auto changed_status = RunHloPass(
        SubByteNormalization(SubByteNormalization::SET_ELEMENT_SIZE), module);
    TF_ASSERT_OK(changed_status.status());
    EXPECT_EQ(change_expected, changed_status.value());
  }

  void RunRemoveElementSize(HloModule* module, bool change_expected) {
    auto changed_status = RunHloPass(
        SubByteNormalization(SubByteNormalization::REMOVE_ELEMENT_SIZE),
        module);
    TF_ASSERT_OK(changed_status.status());
    EXPECT_EQ(change_expected, changed_status.value());
  }
};

TEST_F(SubByteNormalizationTest, SetElementSizeForS4) {
  // Test that S4 types get element_size_in_bits set to 4.
  constexpr std::string_view kModuleStr = R"(
    HloModule SetElementSizeForS4

    ENTRY main {
      p0 = s4[10] parameter(0)
      ROOT neg = s4[10] negate(p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RunSetElementSize(module.get(), /*change_expected=*/true);

  // Verify element_size_in_bits is set to 4 for S4 type.
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(root->shape().has_layout());
  EXPECT_EQ(root->shape().layout().element_size_in_bits(), 4);
}

TEST_F(SubByteNormalizationTest, SetElementSizeForU4) {
  // Test that U4 types get element_size_in_bits set to 4.
  constexpr std::string_view kModuleStr = R"(
    HloModule SetElementSizeForU4

    ENTRY main {
      p0 = u4[10] parameter(0)
      ROOT copy = u4[10] copy(p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RunSetElementSize(module.get(), /*change_expected=*/true);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(root->shape().has_layout());
  EXPECT_EQ(root->shape().layout().element_size_in_bits(), 4);
}

TEST_F(SubByteNormalizationTest, SetElementSizeForS32DoesNotChange) {
  // Test that S32 types do not get element_size_in_bits set (not sub-byte).
  constexpr std::string_view kModuleStr = R"(
    HloModule SetElementSizeForS32

    ENTRY main {
      p0 = s32[10] parameter(0)
      ROOT neg = s32[10] negate(p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RunSetElementSize(module.get(), /*change_expected=*/false);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(root->shape().has_layout());
  EXPECT_EQ(root->shape().layout().element_size_in_bits(), 0);
}

TEST_F(SubByteNormalizationTest, RemoveElementSize) {
  // Test that element_size_in_bits is removed.
  constexpr std::string_view kModuleStr = R"(
    HloModule RemoveElementSize

    ENTRY main {
      p0 = s4[10]{0:E(4)} parameter(0)
      ROOT neg = s4[10]{0:E(4)} negate(p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // First verify element_size_in_bits is set.
  const HloInstruction* root_before =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(root_before->shape().layout().element_size_in_bits(), 4);

  RunRemoveElementSize(module.get(), /*change_expected=*/true);

  // Verify element_size_in_bits is now 0.
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->shape().layout().element_size_in_bits(), 0);
}

TEST_F(SubByteNormalizationTest, TupleShape) {
  // Test that tuple shapes are handled recursively.
  constexpr std::string_view kModuleStr = R"(
    HloModule TupleShape

    ENTRY main {
      p0 = s4[10] parameter(0)
      p1 = s32[10] parameter(1)
      ROOT tuple = (s4[10], s32[10]) tuple(p0, p1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RunSetElementSize(module.get(), /*change_expected=*/true);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(root->shape().IsTuple());
  // S4 element should have element_size_in_bits = 4.
  EXPECT_EQ(root->shape().tuple_shapes(0).layout().element_size_in_bits(), 4);
  // S32 element should have element_size_in_bits = 0 (not sub-byte).
  EXPECT_EQ(root->shape().tuple_shapes(1).layout().element_size_in_bits(), 0);
}

TEST_F(SubByteNormalizationTest, PassName) {
  // Test that the pass names are correct.
  SubByteNormalization set_pass(SubByteNormalization::SET_ELEMENT_SIZE);
  EXPECT_EQ(set_pass.name(), "sub-byte-size-setter");

  SubByteNormalization remove_pass(SubByteNormalization::REMOVE_ELEMENT_SIZE);
  EXPECT_EQ(remove_pass.name(), "sub-byte-size-removal");
}

}  // namespace
}  // namespace zkx
