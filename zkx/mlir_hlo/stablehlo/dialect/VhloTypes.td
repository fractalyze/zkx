/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
Copyright 2022 The StableHLO Authors.
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

#ifndef ZKX_MLIR_HLO_STABLEHLO_DIALECT_VHLO_TYPES_TD
#define ZKX_MLIR_HLO_STABLEHLO_DIALECT_VHLO_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "zkx/mlir_hlo/stablehlo/dialect/VhloBase.td"
include "zkx/mlir_hlo/stablehlo/dialect/VhloDialect.td"

def VHLO_VersionedTypeInterface : TypeInterface<"VersionedTypeInterface"> {
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

class VHLO_TypeDef<string cppName, string name, string minVersion, string maxVersion>
  : TypeDef<VHLO_Dialect, cppName, [VHLO_VersionedTypeInterface]> {
  let mnemonic = name;
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

// Corresponds to BooleanType from the StableHLO spec.
def VHLO_BooleanV1 : VHLO_TypeDef<"BooleanV1", "bool_v1", "0.0.1", "current">;

// Corresponds to FunctionType from the StableHLO spec.
def VHLO_FunctionV1 : VHLO_TypeDef<"FunctionV1", "func_v1", "0.0.1", "current"> {
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$inputs,
    ArrayRefParameter<"mlir::Type">:$outputs
  );
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult FunctionV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
        ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> outputs) {
      if (!allFromVhlo(inputs) || !allFromVhlo(outputs))
        return errFn() << "expected VHLO types";
      return success();
    }
  }];
  let assemblyFormat = "`<` `(` custom<TypeArray>($inputs) `)` `->` custom<TypeArray>($outputs) `>`";
}

// TODO(#8): Index is not part of the StableHLO spec.
// At the moment, it is used to represent values participating in shape
// computations, and we're planning to look into it as part of the work on the
// dynamism RFC.
def VHLO_IndexV1 : VHLO_TypeDef<"IndexV1", "index_v1", "0.0.1", "current">;

// Corresponds to the 'si2' IntegerType from the StableHLO spec.
def VHLO_IntegerSI2V1 : VHLO_TypeDef<"IntegerSI2V1", "i2_v1", "1.2.0", "current">;

// Corresponds to the 'si4' IntegerType from the StableHLO spec.
def VHLO_IntegerSI4V1 : VHLO_TypeDef<"IntegerSI4V1", "i4_v1", "0.0.1", "current">;

// Corresponds to the 'si8' IntegerType from the StableHLO spec.
def VHLO_IntegerSI8V1 : VHLO_TypeDef<"IntegerSI8V1", "i8_v1", "0.0.1", "current">;

// Corresponds to the 'si16' IntegerType from the StableHLO spec.
def VHLO_IntegerSI16V1 : VHLO_TypeDef<"IntegerSI16V1", "i16_v1", "0.0.1", "current">;

// Corresponds to the 'si32' IntegerType from the StableHLO spec.
def VHLO_IntegerSI32V1 : VHLO_TypeDef<"IntegerSI32V1", "i32_v1", "0.0.1", "current">;

// Corresponds to the 'si64' IntegerType from the StableHLO spec.
def VHLO_IntegerSI64V1 : VHLO_TypeDef<"IntegerSI64V1", "i64_v1", "0.0.1", "current">;

// Corresponds to the 'ui2' IntegerType from the StableHLO spec.
def VHLO_IntegerUI2V1 : VHLO_TypeDef<"IntegerUI2V1", "ui2_v1", "1.2.0", "current">;

// Corresponds to the 'ui4' IntegerType from the StableHLO spec.
def VHLO_IntegerUI4V1 : VHLO_TypeDef<"IntegerUI4V1", "ui4_v1", "0.0.1", "current">;

// Corresponds to the 'ui8' IntegerType from the StableHLO spec.
def VHLO_IntegerUI8V1 : VHLO_TypeDef<"IntegerUI8V1", "ui8_v1", "0.0.1", "current">;

// Corresponds to the 'ui16' IntegerType from the StableHLO spec.
def VHLO_IntegerUI16V1 : VHLO_TypeDef<"IntegerUI16V1", "ui16_v1", "0.0.1", "current">;

// Corresponds to the 'ui32' IntegerType from the StableHLO spec.
def VHLO_IntegerUI32V1 : VHLO_TypeDef<"IntegerUI32V1", "ui32_v1", "0.0.1", "current">;

// Corresponds to the 'ui64' IntegerType from the StableHLO spec.
def VHLO_IntegerUI64V1 : VHLO_TypeDef<"IntegerUI64V1", "ui64_v1", "0.0.1", "current">;

// The nonetype is used for optional values which have implementation specific
// default values, for attributes which must be specified as None, use TypedAttr
// of NoneType.
def VHLO_NoneTypeV1 : VHLO_TypeDef<"NoneV1", "none_v1", "1.6.0", "current">;

// Corresponds to TensorType from the StableHLO spec.
// TODO(#8): Encoding is not part of the StableHLO spec.
// At the moment, it is used to represent dimension bounds to support bounded
// dynamism, and we're planning to look into it as part of the work on the
// dynamism RFC.
def VHLO_RankedTensorV1 : VHLO_TypeDef<"RankedTensorV1", "tensor_v1", "0.0.1", "current"> {
  let parameters = (ins
    VHLO_Dims:$shape,
    "::mlir::Type":$elementType,
    "::mlir::Attribute":$encoding
  );
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult RankedTensorV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn,
        ArrayRef<int64_t> shape, ::mlir::Type elementType, ::mlir::Attribute encoding) {
      if (!isFromVhlo(elementType) || (encoding && !isFromVhlo(encoding)))
        return errFn() << "expected VHLO type or attribute";
      return success();
    }
  }];
  let assemblyFormat = "`<` custom<Shape>($shape) `` $elementType `` custom<Encoding>($encoding) `>`";
}

// Corresponds to TokenType from the StableHLO spec.
def VHLO_TokenV1 : VHLO_TypeDef<"TokenV1", "token_v1", "0.0.1", "current">;

// Corresponds to TupleType from the StableHLO spec.
def VHLO_TupleV1 : VHLO_TypeDef<"TupleV1", "tuple_v1", "0.0.1", "current"> {
  let parameters = (ins ArrayRefParameter<"::mlir::Type">:$types);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult TupleV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, ArrayRef<mlir::Type> types) {
      if (!allFromVhlo(types)) return errFn() << "expected VHLO types";
      return success();
    }
  }];
  let assemblyFormat = "`<` $types `>`";
}

// TODO(#8): UnrankedTensor is not part of the StableHLO spec.
// At the moment, it is used to represent unranked dynamism, and we will likely
// remove it as part of the work on the dynamism RFC.
def VHLO_UnrankedTensorV1 : VHLO_TypeDef<"UnrankedTensorV1", "unranked_tensor_v1", "0.0.1", "current"> {
  let parameters = (ins "::mlir::Type":$elementType);
  let genVerifyDecl = 1;
  let extraClassDefinition = [{
    LogicalResult UnrankedTensorV1Type::verify(
        llvm::function_ref<mlir::InFlightDiagnostic ()> errFn, mlir::Type elementType) {
      if (!isFromVhlo(elementType)) return errFn() << "expected VHLO type";
      return success();
    }
  }];
  let assemblyFormat = "`<` $elementType `>`";
}

// TODO(#8): Witness is not part of the StableHLO spec.
// At the moment, it is used to represent constraints for dynamic shapes,
// and we're planning to look into it as part of the work on the dynamism RFC.
def VHLO_WitnessV1 : VHLO_TypeDef<"WitnessV1", "witness_v1", "0.0.1", "current">;

#endif  // ZKX_MLIR_HLO_STABLEHLO_DIALECT_VHLO_TYPES_TD
