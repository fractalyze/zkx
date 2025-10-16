/* Copyright 2020 The OpenXLA Authors.

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

#ifndef ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_ENUMS_TD
#define ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_ENUMS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Comparison op definitions.
//===----------------------------------------------------------------------===//

def MHLO_LOOP_FUSION : I32EnumAttrCase<"kLoop", 0>;
def MHLO_INPUT_FUSION : I32EnumAttrCase<"kInput", 1>;
def MHLO_OUTPUT_FUSION : I32EnumAttrCase<"kOutput", 2>;
def MHLO_CUSTOM_FUSION : I32EnumAttrCase<"kCustom", 3>;
def MHLO_FusionKind : I32EnumAttr<"FusionKind", "fusion kind", [
    MHLO_LOOP_FUSION, MHLO_INPUT_FUSION, MHLO_OUTPUT_FUSION, MHLO_CUSTOM_FUSION
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mhlo";
}

def MHLO_FusionKindAttr : EnumAttr<MHLO_Dialect, MHLO_FusionKind, "fusion_kind">;

#endif // ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_ENUMS_TD
