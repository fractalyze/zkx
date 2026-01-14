/* Copyright 2024 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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
#include "zkx/codegen/emitters/computation_partitioner.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#include "zkx/hlo/analysis/indexing_analysis.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/tests/hlo_test_base.h"

namespace zkx::emitters {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class ComputationPartitionerTest : public HloTestBase {
 protected:
  ComputationPartitionerTest() {
    mlir_context_.loadDialect<mlir::func::FuncDialect>();
  }

  mlir::MLIRContext mlir_context_;
};

std::string PrintAndErase(mlir::func::FuncOp func) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << func;
  // Erase the function so we don't leak memory.
  func.erase();
  return out;
}

// clang-format off
// TODO(chokobole): Enable this test. Dependency: HloInstruction::PrintWithCanonicalNameMap
// clang-format on
TEST_F(ComputationPartitionerTest, DISABLED_PartitionDiamonds) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    fused_computation {
      %param = s32[6] parameter(0)
      %slice0.1 = s32[5] slice(s32[6]{0} %param), slice={[0:5]}
      %slice0.2 = s32[5] slice(s32[6]{0} %param), slice={[1:6]}
      %add0 = s32[5] add(s32[5]{0} %slice0.1, s32[5]{0} %slice0.2)
      %slice1.1 = s32[4] slice(s32[5]{0} %add0), slice={[0:4]}
      %slice1.2 = s32[4] slice(s32[5]{0} %add0), slice={[1:5]}
      %add1 = s32[4] add(s32[4]{0} %slice1.1, s32[4]{0} %slice1.2)
      %slice2.1 = s32[3] slice(s32[4]{0} %add1), slice={[0:3]}
      %slice2.2 = s32[3] slice(s32[4]{0} %add1), slice={[1:4]}
      %add2 = s32[3] add(s32[3]{0} %slice2.1, s32[3]{0} %slice2.2)
      %slice3.1 = s32[2] slice(s32[3]{0} %add2), slice={[0:2]}
      %slice3.2 = s32[2] slice(s32[3]{0} %add2), slice={[1:3]}
      ROOT %add3 = s32[2] add(s32[2]{0} %slice3.1, s32[2]{0} %slice3.2)
    })")
                    .value();

  auto* fusion = module->GetComputationWithName("fused_computation");
  ASSERT_NE(fusion, nullptr);
  PartitionedComputation computation(fusion, &mlir_context_);

  constexpr auto kExpected = R"(PartitionedComputation fused_computation:
      SUBGRAPH fused_computation_add3 {
        %slice3.1 = s32[2]{0} slice(s32[3]{0} %add2), slice={[0:2]}
        %slice3.2 = s32[2]{0} slice(s32[3]{0} %add2), slice={[1:3]}
        ROOT %add3 = s32[2]{0} add(s32[2]{0} %slice3.1, s32[2]{0} %slice3.2)
      }
      SUBGRAPH fused_computation_add2 {
        %slice2.1 = s32[3]{0} slice(s32[4]{0} %add1), slice={[0:3]}
        %slice2.2 = s32[3]{0} slice(s32[4]{0} %add1), slice={[1:4]}
        ROOT %add2 = s32[3]{0} add(s32[3]{0} %slice2.1, s32[3]{0} %slice2.2)
      }
      SUBGRAPH fused_computation_add1 {
        %slice1.1 = s32[4]{0} slice(s32[5]{0} %add0), slice={[0:4]}
        %slice1.2 = s32[4]{0} slice(s32[5]{0} %add0), slice={[1:5]}
        ROOT %add1 = s32[4]{0} add(s32[4]{0} %slice1.1, s32[4]{0} %slice1.2)
      }
      SUBGRAPH fused_computation_add0 {
        %slice0.1 = s32[5]{0} slice(s32[6]{0} %param), slice={[0:5]}
        %slice0.2 = s32[5]{0} slice(s32[6]{0} %param), slice={[1:6]}
        ROOT %add0 = s32[5]{0} add(s32[5]{0} %slice0.1, s32[5]{0} %slice0.2)
      }
      SUBGRAPH fused_computation_param {
        ROOT %param = s32[6]{0} parameter(0)
      })";
  EXPECT_EQ(computation.ToString(6), kExpected);
}

TEST_F(ComputationPartitionerTest, SimpleConcatenate) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    fused_computation {
      %param1 = s32[6] parameter(0)
      %param2 = s32[3] parameter(1)
      %neg = s32[6] negate(%param1)
      %abs = s32[3] abs(%param2)
      ROOT %concat = s32[9] concatenate(%neg, %abs), dimensions={0}
    })")
                    .value();

  auto* fusion = module->GetComputationWithName("fused_computation");
  ASSERT_NE(fusion, nullptr);
  PartitionedComputation computation(fusion, &mlir_context_);

  EXPECT_THAT(computation.subgraphs(), SizeIs(1));
}

// clang-format off
// TODO(chokobole): Enable this test. Dependency: HloInstruction::PrintWithCanonicalNameMap
// clang-format on
TEST_F(ComputationPartitionerTest, DISABLED_DiamondConcatenate) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    fused_computation {
      %param1 = s32[6] parameter(0)
      %param2 = s32[6] parameter(1)
      %not = s32[6] not(%param1)
      %add = s32[6] add(%not, %param2)
      %neg = s32[6] negate(%not)
      %abs = s32[6] abs(%add)
      ROOT %concat = s32[12] concatenate(%neg, %abs), dimensions={0}
    })")
                    .value();

  auto* fusion = module->GetComputationWithName("fused_computation");
  ASSERT_NE(fusion, nullptr);
  PartitionedComputation computation(fusion, &mlir_context_);

  constexpr auto kExpected = R"(PartitionedComputation fused_computation:
      SUBGRAPH fused_computation_concat {
        %neg = s32[6]{0} negate(s32[6]{0} %not)
        %param2 = s32[6]{0} parameter(1)
        %add = s32[6]{0} add(s32[6]{0} %log, s32[6]{0} %param2)
        %abs = s32[6]{0} abs(s32[6]{0} %add)
        ROOT %concat = s32[12]{0} concatenate(s32[6]{0} %neg, s32[6]{0} %abs), dimensions={0}
      }
      SUBGRAPH fused_computation_not {
        %param1 = s32[6]{0} parameter(0)
        ROOT %not = s32[6]{0} not(s32[6]{0} %param1)
      })";
  EXPECT_EQ(computation.ToString(6), kExpected);
}

TEST_F(ComputationPartitionerTest, TupleRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    fused_computation {
      %p0 = s32[6] parameter(0)
      %p1 = s32[6] parameter(1)
      %add = s32[6] add(p0, p1)
      %sub = s32[6] subtract(p0, p1)
      ROOT %root = (s32[6], s32[6]) tuple(%add, %sub)
    })")
                    .value();

  auto* fusion = module->GetComputationWithName("fused_computation");
  ASSERT_NE(fusion, nullptr);
  PartitionedComputation computation(fusion, &mlir_context_);
  // We don't analyze the actual indexes of the tuple, so we assume %add and
  // %sub have different indexing. That's why the parameters end up in separate
  // functions.
  constexpr auto kExpected = R"(PartitionedComputation fused_computation:
      SUBGRAPH fused_computation_root {
        %add = s32[6]{0} add(s32[6]{0} %p0, s32[6]{0} %p1)
        %sub = s32[6]{0} subtract(s32[6]{0} %p0, s32[6]{0} %p1)
        ROOT %root = (s32[6]{0}, s32[6]{0}) tuple(s32[6]{0} %add, s32[6]{0} %sub)
      }
      SUBGRAPH fused_computation_p1 {
        ROOT %p1 = s32[6]{0} parameter(1)
      }
      SUBGRAPH fused_computation_p0 {
        ROOT %p0 = s32[6]{0} parameter(0)
      })";
  EXPECT_EQ(computation.ToString(6), kExpected);
}

TEST_F(ComputationPartitionerTest, Epilogue) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT add = s32[] add(p0, p1)
    }

    fused_computation {
      p0 = s32[4] parameter(0)
      c0 = s32[] constant(0)
      reduce = s32[] reduce(p0, c0), dimensions={0}, to_apply=add
      bitcast = s32[1] bitcast(reduce)
      abs = s32[1] abs(bitcast)
      not = s32[1] not(abs)
      sign = s32[1] sign(bitcast)
      ROOT tuple = (s32[1], s32[1]) tuple(not, sign)
    })")
                    .value();

  auto* fused_computation = module->GetComputationWithName("fused_computation");
  EpilogueSpecification epilogue{
      /*heroes=*/{fused_computation->GetInstructionWithName("reduce")},
      /*roots=*/
      {fused_computation->GetInstructionWithName("not"),
       fused_computation->GetInstructionWithName("sign")},
      /*index_ranges=*/{1, 42},
      {CreateIdentityMap(
          fused_computation->root_instruction()->shape().tuple_shapes(0),
          &mlir_context_)}};
  PartitionedComputations fusion(fused_computation, &mlir_context_, {epilogue});

  mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(&mlir_context_),
                                     &mlir_context_);
  EXPECT_EQ(
      PrintAndErase(
          CreateSubgraphMlirFunction(fusion.epilogues().front(), builder)),
      "func.func private @fused_computation__epilogue__not_sign(tensor<4xi32>, "
      "index {zkx.range = [0 : index, 0 : index]}, "
      "index {zkx.range = [0 : index, 41 : index]}, "
      "i32) -> (i32, i32)");
}

TEST_F(ComputationPartitionerTest, TransposeAsRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    fused_computation {
      %p0 = s32[64, 32] parameter(0)
      %p1 = s32[64, 32] parameter(1)
      %add = s32[64, 32] add(p0, p1)
      %transpose = s32[32, 64] transpose(%add), dimensions={1, 0}
      %neg = s32[32, 64] negate(%transpose)
      ROOT %root = s32[32, 64] sign(%neg)
    })")
                    .value();

  auto* fusion = module->GetComputationWithName("fused_computation");
  ASSERT_NE(fusion, nullptr);
  PartitionedComputation computation(
      fusion, &mlir_context_, [](const HloInstruction* instr) {
        return instr->opcode() == HloOpcode::kTranspose;
      });
  ASSERT_THAT(computation.subgraphs(), SizeIs(2));
  EXPECT_THAT(computation.GetRootSubgraph().roots, SizeIs(1));
  EXPECT_THAT(computation.GetRootSubgraph().instructions, SizeIs(2));
}

TEST_F(ComputationPartitionerTest, PartiallyMergable) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    fused_computation {
      %p0 = s32[10,10] parameter(0)
      %p1 = s32[10,10] parameter(1)
      %add = s32[10,10] add(%p0, %p1)
      %transpose = s32[10,10] transpose(%add), dimensions={1,0}
      ROOT %sub = s32[10,10] subtract(%add, %transpose)
    })")
                    .value();

  auto* fusion = module->GetComputationWithName("fused_computation");
  ASSERT_NE(fusion, nullptr);
  PartitionedComputation computation(fusion, &mlir_context_);

  auto transpose = fusion->GetInstructionWithName("transpose");
  auto sub = fusion->GetInstructionWithName("sub");

  ASSERT_THAT(computation.subgraphs(), SizeIs(2));
  EXPECT_THAT(computation.GetRootSubgraph().instructions,
              UnorderedElementsAre(transpose, sub));
}

TEST_F(ComputationPartitionerTest, SubgraphSignatures) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      %p0 = s32[] parameter(0)
      %p1 = s32[] parameter(1)
      ROOT %add = s32[] add(%p0, %p1)
    }

    fusion {
      %p0 = s32[10,10]{0,1} parameter(0)
      %p1 = s32[10,10]{1,0} parameter(1)
      %c0 = s32[] constant(2)
      %bc = s32[10,10]{0,1} bitcast(%p1)
      %add = s32[10,10] add(%p0, %bc)
      ROOT %reduce = s32[10] reduce(%add, %c0), dimensions={1}, to_apply=add
    }

    ENTRY main {
      %p0 = s32[10,10] parameter(0)
      %p1 = s32[10,10] parameter(1)
      ROOT %fusion = s32[10] fusion(%p0, %p1), kind=kLoop, calls=fusion
    })")
                    .value();

  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(&context), &context);

  PartitionedComputation fusion(module->GetComputationWithName("fusion"),
                                &mlir_context_);
  EXPECT_EQ(
      PrintAndErase(
          CreateSubgraphMlirFunction(fusion.GetRootSubgraph(), builder)),
      "func.func private @fusion_reduce(tensor<10x10xi32, dense<[0, 1]> : "
      "tensor<2xi64>>, tensor<10x10xi32>, index {zkx.range = [0 : index, 9 : "
      "index]}) -> i32");

  PartitionedComputation add(module->GetComputationWithName("add"),
                             &mlir_context_);
  EXPECT_EQ(
      PrintAndErase(CreateSubgraphMlirFunction(add.GetRootSubgraph(), builder)),
      "func.func private @add_add(i32, i32) -> i32");
}

}  // namespace
}  // namespace zkx::emitters
