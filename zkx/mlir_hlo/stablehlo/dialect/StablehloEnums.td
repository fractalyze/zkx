/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLO_ENUMS_TD
#define ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLO_ENUMS_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Comparison op definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA ComparisonDirection enum.
def STABLEHLO_COMPARISON_DIRECTION_EQ : I32EnumAttrCase<"EQ", 0>;
def STABLEHLO_COMPARISON_DIRECTION_NE : I32EnumAttrCase<"NE", 1>;
def STABLEHLO_COMPARISON_DIRECTION_GE : I32EnumAttrCase<"GE", 2>;
def STABLEHLO_COMPARISON_DIRECTION_GT : I32EnumAttrCase<"GT", 3>;
def STABLEHLO_COMPARISON_DIRECTION_LE : I32EnumAttrCase<"LE", 4>;
def STABLEHLO_COMPARISON_DIRECTION_LT : I32EnumAttrCase<"LT", 5>;

def StableHLO_ComparisonDirection : I32EnumAttr<"ComparisonDirection",
    "Which comparison operation to perform.",
    [
      STABLEHLO_COMPARISON_DIRECTION_EQ,
      STABLEHLO_COMPARISON_DIRECTION_NE,
      STABLEHLO_COMPARISON_DIRECTION_GE,
      STABLEHLO_COMPARISON_DIRECTION_GT,
      STABLEHLO_COMPARISON_DIRECTION_LE,
      STABLEHLO_COMPARISON_DIRECTION_LT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::stablehlo";
}

def StableHLO_ComparisonDirectionAttr : EnumAttr<StableHLO_Dialect, StableHLO_ComparisonDirection, "comparison_direction">;

#endif // ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLO_ENUMS_TD
