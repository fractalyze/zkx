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

#include "zkx/service/gpu/model/gpu_hlo_cost_analysis.h"

#include <cstdint>

#include "gtest/gtest.h"

#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_module.h"
#include "zkx/shape_util.h"
#include "zkx/tests/hlo_test_base.h"

namespace zkx {
namespace gpu {
namespace {

class GpuHloCostAnalysisTest : public HloTestBase {
 protected:
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [](const Shape& shape) {
      return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/8);
    };
  }
};

TEST_F(GpuHloCostAnalysisTest, Add) {
  const char* hlo_text = R"(
    HloModule test_module

    ENTRY main {
      p0 = s32[10] parameter(0)
      p1 = s32[10] parameter(1)
      ROOT add = s32[10] add(p0, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloCostAnalysis::Options options{ShapeSizeBytesFunction()};
  GpuHloCostAnalysis analysis(options);
  TF_ASSERT_OK(
      module->entry_computation()->root_instruction()->Accept(&analysis));

  // GPU cost analysis uses a default of 3 cycles per elementwise op
  // 10 elements * 3 cycles = 30
  EXPECT_EQ(analysis.flop_count(), 30);
}

TEST_F(GpuHloCostAnalysisTest, Reduce) {
  const char* hlo_text = R"(
    HloModule test_module

    add_computation {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY main {
      p0 = s32[100, 10] parameter(0)
      init = s32[] constant(0)
      ROOT reduce = s32[100] reduce(p0, init), dimensions={1}, to_apply=add_computation
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloCostAnalysis::Options options{ShapeSizeBytesFunction()};
  GpuHloCostAnalysis analysis(options);
  TF_ASSERT_OK(
      module->entry_computation()->root_instruction()->Accept(&analysis));

  // Reduce over dimension of size 10: 100 output elements, each requires 9
  // adds. GPU uses 3 cycles per add = 100 * 9 * 3 = 2700 FLOPs
  EXPECT_EQ(analysis.flop_count(), 2700);
}

TEST_F(GpuHloCostAnalysisTest, IrSize) {
  const char* hlo_text = R"(
    HloModule test_module

    ENTRY main {
      p0 = s32[10] parameter(0)
      p1 = s32[10] parameter(1)
      ROOT add = s32[10] add(p0, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloCostAnalysis::Options options{ShapeSizeBytesFunction()};
  GpuHloCostAnalysis analysis(options);
  TF_ASSERT_OK(
      module->entry_computation()->root_instruction()->Accept(&analysis));

  // IR size should be set for each instruction
  const HloInstruction* add = module->entry_computation()->root_instruction();
  EXPECT_EQ(analysis.IrSize(*add), 1);
}

TEST_F(GpuHloCostAnalysisTest, WholeModule) {
  const char* hlo_text = R"(
    HloModule test_module

    ENTRY main {
      p0 = s32[10] parameter(0)
      p1 = s32[10] parameter(1)
      add = s32[10] add(p0, p1)
      ROOT mul = s32[10] multiply(add, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  HloCostAnalysis::Options options{ShapeSizeBytesFunction()};
  GpuHloCostAnalysis analysis(options);
  TF_ASSERT_OK(module->entry_computation()->Accept(&analysis));

  // 10 adds + 10 multiplies = 20 ops, GPU uses 3 cycles per op = 60 FLOPs
  EXPECT_EQ(analysis.flop_count(), 60);
}

}  // namespace
}  // namespace gpu
}  // namespace zkx
