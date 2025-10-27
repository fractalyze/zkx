/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef ZKX_MLIR_HLO_STABLEHLO_DIALECT_VHLO_ENUMS_TD
#define ZKX_MLIR_HLO_STABLEHLO_DIALECT_VHLO_ENUMS_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"
include "zkx/mlir_hlo/stablehlo/dialect/VhloBase.td"
include "zkx/mlir_hlo/stablehlo/dialect/VhloDialect.td"
include "mlir/IR/AttrTypeBase.td"

def VHLO_VersionedAttrInterface : AttrInterface<"VersionedAttrInterface"> {
  let cppNamespace = "::mlir::vhlo";
  let methods = [
    InterfaceMethod<
      "Returns the minimum version of the VHLO dialect an attribute is supported in.",
      "mlir::vhlo::Version", "getMinVersion">,
    InterfaceMethod<
      "Returns the maximum version (inclusive) of the VHLO dialect an attribute is supported in.",
      "mlir::vhlo::Version", "getMaxVersion">,
  ];
}

class VHLO_I32EnumAttr<string name, list<I32EnumAttrCase> cases> :
    I32EnumAttr<name, name, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::vhlo";
}

class VHLO_EnumAttr<EnumAttrInfo enumInfo, string name, string minVersion, string maxVersion>
  : EnumAttr<VHLO_Dialect, enumInfo, name, [VHLO_VersionedAttrInterface]> {
  let extraClassDeclaration = [{
    mlir::vhlo::Version getMinVersion() {
      return mlir::vhlo::Version(}] # !subst(".", ", ", minVersion) # [{);
    }
    mlir::vhlo::Version getMaxVersion() {
      }] # !if(
        !eq(maxVersion, "current"),
        [{ return mlir::vhlo::Version::getCurrentVersion(); }],
        [{ return mlir::vhlo::Version("}] # !subst(".", ", ", maxVersion) # [{"); }]
      ) # [{
    }
  }];
}

//===----------------------------------------------------------------------===//
// ComparisonDirection
//===----------------------------------------------------------------------===//

def VHLO_COMPARISON_DIRECTION_V1_EQ : I32EnumAttrCase<"EQ", 0>;
def VHLO_COMPARISON_DIRECTION_V1_NE : I32EnumAttrCase<"NE", 1>;
def VHLO_COMPARISON_DIRECTION_V1_GE : I32EnumAttrCase<"GE", 2>;
def VHLO_COMPARISON_DIRECTION_V1_GT : I32EnumAttrCase<"GT", 3>;
def VHLO_COMPARISON_DIRECTION_V1_LE : I32EnumAttrCase<"LE", 4>;
def VHLO_COMPARISON_DIRECTION_V1_LT : I32EnumAttrCase<"LT", 5>;

def VHLO_ComparisonDirectionV1 : VHLO_I32EnumAttr<"ComparisonDirectionV1", [
  VHLO_COMPARISON_DIRECTION_V1_EQ,
  VHLO_COMPARISON_DIRECTION_V1_NE,
  VHLO_COMPARISON_DIRECTION_V1_GE,
  VHLO_COMPARISON_DIRECTION_V1_GT,
  VHLO_COMPARISON_DIRECTION_V1_LE,
  VHLO_COMPARISON_DIRECTION_V1_LT
]> {}

def VHLO_ComparisonDirectionAttrV1
  : VHLO_EnumAttr<VHLO_ComparisonDirectionV1, "comparison_direction_v1", "0.0.1", "current">;

#endif // ZKX_MLIR_HLO_STABLEHLO_DIALECT_VHLO_ENUMS_TD
