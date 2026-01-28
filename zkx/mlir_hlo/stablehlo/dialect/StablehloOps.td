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

#ifndef ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLO_OPS_TD
#define ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLO_OPS_TD

include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "zkx/mlir_hlo/stablehlo/dialect/Base.td"

def StableHLO_Dialect : Dialect {
  let name = "stablehlo";
  let cppNamespace = "::mlir::stablehlo";

  let description = [{
    StableHLO is an operation set that expresses ZK computations. It has been
    originally bootstrapped from the MHLO dialect and enhances it with additional
    functionality, including serialization and versioning, to be used as
    a portability layer between ZK frameworks and ZK compilers.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

class StableHLO_Op<string mnemonic, list<Trait> traits = []> :
    Op<StableHLO_Dialect, mnemonic, traits> {
  string commonClassDeclaration = [{
    // Relax the strict default implementation with one that allows
    // for StableHLO-specific differences.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
  let extraClassDeclaration = commonClassDeclaration;
}

include "zkx/mlir_hlo/stablehlo/dialect/StablehloEnums.td"
include "zkx/mlir_hlo/stablehlo/dialect/StablehloAttrs.td"
include "zkx/mlir_hlo/stablehlo/dialect/StablehloTypes.td"

class StableHLO_ShapedInterfaceOp<string mnemonic, list<Trait> traits> :
    StableHLO_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
    ["reifyReturnTypeShapes"]>]> {}

//===----------------------------------------------------------------------===//
// StableHLO nullary op definitions.
//===----------------------------------------------------------------------===//

def StableHLO_ConstantOp : StableHLO_Op<"constant",
    [ConstantLike, Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Constant operation";
  let description = [{
    Produces an `output` tensor from a constant `value`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant

    Example:
    ```mlir
    %output = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xu32>
    ```
  }];
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    HLO_StaticShapeTensor:$output
  );

  let builders = [
    OpBuilder<(ins "Type":$type, "Attribute":$value)>];

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

def StableHLO_IotaOp : StableHLO_Op<"iota", [Pure]> {
  let summary = "Iota operation";
  let description = [{
    Fills an `output` tensor with values in increasing order starting from zero
    along the `iota_dimension` dimension.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota

    Example:
    ```mlir
    %output = stablehlo.iota dim = 0 : tensor<4x5xi32>
    ```
  }];
  let arguments = (ins
    ConfinedAttr<I64Attr, [IntNonNegative]>:$iota_dimension /*iota_c1*/
  );

  let results = (outs HLO_StaticShapeIntTensor:$output);

  let hasVerifier = 1;

  let assemblyFormat = "`dim` `=` $iota_dimension attr-dict `:` type($output)";
}

def StableHLO_DynamicIotaOp: StableHLO_ShapedInterfaceOp<"dynamic_iota", [ConditionallySpeculatable, NoMemoryEffect]> {
  let summary = "DynamicIota operation";
  let description = [{
    This operation is functionally identical to
    [iota](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#iota)
    op, but the result shape is specified dynamically via `output_shape`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_iota

    Example:
    ```mlir
    %output_shape = stablehlo.constant dense<[4, 5]> : tensor<2xi64>
    %0 = stablehlo.dynamic_iota %output_shape, dim = 0 : (tensor<2xi64>) -> tensor<4x5xi64>
    ```
  }];

  let arguments = (ins
    HLO_StaticDimensionTensor:$output_shape /*dynamic_iota_i1*/,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$iota_dimension /*dynamic_iota_c1, dynamic_iota_i2*/
  );
  let results = (outs HLO_Tensor:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $output_shape `,` `dim` `=` $iota_dimension attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_CreateTokenOp : StableHLO_Op<"create_token", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "CreateToken operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as AfterAllOp with 0 inputs:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#after_all

    Example:
    ```mlir
    %output = stablehlo.create_token : !stablehlo.token
    ```
  }];

  let results = (outs HLO_Token:$output);

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// StableHLO unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class StableHLO_UnaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type OperandType, Type ResultType = OperandType> : StableHLO_Op<mnemonic,
    traits # [Elementwise, InferShapedTypeOpInterface, SameOperandsAndResultShape,
              HLO_SpeculatableIfStaticDimInOutputIsStaticInInput, NoMemoryEffect]> {
  let arguments = (ins OperandType:$operand);
  let results = (outs ResultType:$result);
  let extraClassDeclaration = commonClassDeclaration # [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                operands.front(),
                                                &reifiedReturnShapes);
    }
  }];

  let assemblyFormat = [{
    $operand attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

def StableHLO_AbsOp: StableHLO_UnaryElementwiseOp<"abs",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>],
     RankedTensorOf<[HLO_SInt] /* abs_i1 */>,
     RankedTensorOf<[HLO_SInt]>> {
  let summary = "Abs operation";
  let description = [{
    Performs element-wise abs operation on `operand` tensor and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs

    Example:
    ```mlir
    %result = stablehlo.abs %operand : tensor<3xi32>
    ```
  }];
}

def StableHLO_ConvertOp : StableHLO_UnaryElementwiseOp<"convert",
    [SameOperandsAndResultShape /*convert_c1*/], HLO_Tensor> { /*convert_i1*/
  let summary = "Convert operation";
  let description = [{
    Performs an element-wise conversion from one element type to another on
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convert

    Example:
    ```mlir
    %result = stablehlo.convert %operand : (tensor<3xi64>) -> tensor<3xu64>
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$operand, "Type":$result_element_ty)>];
}

def StableHLO_ClzOp: StableHLO_UnaryElementwiseOp<"count_leading_zeros",
    [HLO_CompatibleOperandsAndResultType /*count_leading_zeros_c1*/],
     HLO_IntTensor /*count_leading_zeros_i1*/> { /*count_leading_zeros_c1*/
  let summary = "Clz operation";
  let description = [{
    Performs element-wise count of the number of leading zero bits in the
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#count_leading_zeros

    Example:
    ```mlir
    %result = stablehlo.count_leading_zeros %operand : tensor<2x2xi64>
    ```
  }];
}

def StableHLO_NotOp: StableHLO_UnaryElementwiseOp<"not",
    [HLO_CompatibleOperandsAndResultType], HLO_PredOrIntTensor> {
  let summary = "Not operation";
  let description = [{
    Performs element-wise NOT of tensor `operand` of type integer and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#not

    Example:
    ```mlir
    %result = stablehlo.not %operand : tensor<5x3x1xi1>
    ```
  }];
}

def StableHLO_NegOp: StableHLO_UnaryElementwiseOp<"negate",
    [HLO_CompatibleOperandsAndResultType], HLO_IntTensor> {
  let summary = "Neg operation";
  let description = [{
    Performs element-wise negation of `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#negate

    Example:
    ```mlir
    %result = stablehlo.negate %operand : tensor<2x3xi32>
    ```
  }];
}

def StableHLO_PopulationCountOp: StableHLO_UnaryElementwiseOp<"popcnt",
    [HLO_CompatibleOperandsAndResultType /*popcnt_c1*/],
    HLO_IntTensor /*popcnt_i1*/> { /*popcnt_c1*/
  let summary = "PopulationCount operation";
  let description = [{
    Performs element-wise count of the number of bits set in the `operand`
    tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#popcnt

    Example:
    ```mlir
    %result = stablehlo.popcnt %operand : tensor<4xi64>
    ```
  }];
}

def StableHLO_SignOp: StableHLO_UnaryElementwiseOp<"sign",
    [HLO_CompatibleOperandsAndResultType /*sign_c1*/],
    RankedTensorOf<[HLO_SInt]> /*sign_i1*/> { /*sign_c1*/
  let summary = "Sign operation";
  let description = [{
    Returns the sign of the `operand` element-wise and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sign

    Example:
    ```mlir
    %result = stablehlo.sign %operand : tensor<5xi64>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StableHLO binary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations

class StableHLO_BinaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type OperandType = HLO_Tensor, Type ResultType = OperandType> :
    StableHLO_Op<mnemonic, traits # [InferShapedTypeOpInterface,
    SameOperandsAndResultShape, Elementwise,
    HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect]> {
  let arguments = (ins
    OperandType:$lhs,
    OperandType:$rhs
  );

  string binaryElementwiseOpCommonClassDeclaration = commonClassDeclaration # [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                 operands.front(),
                                                 &reifiedReturnShapes);
    }
  }];

  let extraClassDeclaration = binaryElementwiseOpCommonClassDeclaration;

  let results = (outs ResultType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def StableHLO_AddOp : StableHLO_BinaryElementwiseOp<"add", [HLO_Commutative,
      InferTypeOpInterface,
      DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>],
      HLO_Tensor> {
  let summary = "Add operation";
  let description = [{
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add

    Example:
    ```mlir
    %result = stablehlo.add %lhs, %rhs : tensor<2x2xi32>
    ```
  }];

  let extraClassDeclaration = binaryElementwiseOpCommonClassDeclaration # [{
    static LogicalResult inferReturnTypes(
        MLIRContext * /*context*/, std::optional<Location> location,
        ValueRange operands, DictionaryAttr /*attributes*/,
        OpaqueProperties /*properties*/, RegionRange /*regions*/,
        SmallVectorImpl<Type> &inferredReturnTypes) {
      if (operands.empty())
        return emitOptionalError(
            location,
            "Expected non-empty operands for AddOp::inferReturnTypes");

      auto inferredTypeOrErr =
          mlir::hlo::inferMostSpecificType(location, operands.getTypes());
      if (failed(inferredTypeOrErr)) return failure();
      inferredReturnTypes.emplace_back(*inferredTypeOrErr);
      return success();
    }
  }];

  let hasVerifier = 1;

}

def StableHLO_DivOp : StableHLO_BinaryElementwiseOp<"divide",
    [HLO_CompatibleOperandsAndResultType /* div_c1 */],
    HLO_IntTensor /* div_i1, div_i2 */> {
  let summary = "Div operation";
  let description = [{
    Performs element-wise division of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#divide

    Example:
    ```mlir
    %result = stablehlo.divide %lhs, %rhs : tensor<4xi32>
    ```
  }];
}

def StableHLO_MaxOp : StableHLO_BinaryElementwiseOp<"maximum",
      [HLO_Commutative, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Max operation";
  let description = [{
    Performs element-wise max operation on tensors `lhs` and `rhs` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#maximum

    Example:
    ```mlir
    %result = stablehlo.maximum %lhs, %rhs : tensor<4xi32>
    ```
  }];
}

def StableHLO_MinOp : StableHLO_BinaryElementwiseOp<"minimum",
      [HLO_Commutative, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Min operation";
  let description = [{
    Performs element-wise min operation on tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#minimum

    Example:
    ```mlir
    %result = stablehlo.minimum %lhs, %rhs : tensor<4xi32>
    ```
  }];
}

def StableHLO_MulOp : StableHLO_BinaryElementwiseOp<"multiply",
      [HLO_Commutative, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Mul operation";
  let description = [{
    Performs element-wise product of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#multiply

    Example:
    ```mlir
    %result = stablehlo.multiply %lhs, %rhs : tensor<2xi32>
    ```
  }];
}

def StableHLO_PowOp : StableHLO_BinaryElementwiseOp<"power",
      [HLO_CompatibleOperandsAndResultType /* pow_c1 */],
      HLO_IntTensor /* pow_i1, pow_i2 */> {
  let summary = "Power operation";
  let description = [{
    Performs element-wise exponentiation of `lhs` tensor by `rhs` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#power

    Example:
    ```mlir
    %result = stablehlo.power %lhs, %rhs : tensor<6xi64>
    ```
  }];
}

def StableHLO_RemOp : StableHLO_BinaryElementwiseOp<"remainder",
      [HLO_CompatibleOperandsAndResultType /*remainder_c1*/],
      HLO_IntTensor /*remainder_i1, remainder_i2*/> {
  let summary = "Rem operation";
  let description = [{
    Performs element-wise remainder of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#remainder

    Example:
    ```mlir
    %result = stablehlo.remainder %lhs, %rhs : tensor<4xi64>
    ```
  }];
}

def StableHLO_ShiftLeftOp : StableHLO_BinaryElementwiseOp<"shift_left",
      [HLO_CompatibleOperandsAndResultType /*shift_left_c1*/],
      HLO_IntTensor /*shift_left_i1, shift_left_i2*/> { /*shift_left_c1*/
  let summary = "ShiftLeft operation";
  let description = [{
    Performs element-wise left-shift operation on the `lhs` tensor by `rhs`
    number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_left

    Example:
    ```mlir
    %result = stablehlo.shift_left %lhs, %rhs : tensor<3xi64>
    ```
  }];
}

def StableHLO_ShiftRightArithmeticOp : StableHLO_BinaryElementwiseOp<"shift_right_arithmetic",
      [HLO_CompatibleOperandsAndResultType /*shift_right_arithmetic_c1*/],
      HLO_IntTensor /*shift_right_arithmetic_i1, shift_right_arithmetic_i2*/> { /*shift_right_arithmetic_c1*/
  let summary = "ShiftRightArithmetic operation";
  let description = [{
    Performs element-wise arithmetic right-shift operation on the `lhs` tensor
    by `rhs` number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_arithmetic

    Example:
    ```mlir
    %result = stablehlo.shift_right_arithmetic %lhs, %rhs : tensor<3xi64>
    ```
  }];
}

def StableHLO_ShiftRightLogicalOp : StableHLO_BinaryElementwiseOp<"shift_right_logical",
      [HLO_CompatibleOperandsAndResultType /*shift_right_logical_c1*/],
      HLO_IntTensor /*shift_right_logical_i1, shift_right_logical_i2*/> { /*shift_right_logical_c1*/
  let summary = "ShiftRightLogical operation";
  let description = [{
    Performs element-wise logical right-shift operation on the `lhs` tensor by
    `rhs` number of bits and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#shift_right_logical

    Example:
    ```mlir
    %result = stablehlo.shift_right_logical %lhs, %rhs : tensor<3xi64>
    ```
  }];
}

def StableHLO_SubtractOp : StableHLO_BinaryElementwiseOp<"subtract",
      [HLO_CompatibleOperandsAndResultType], HLO_IntOrFieldTensor> {
  let summary = "Subtract operation";
  let description = [{
    Performs element-wise subtraction of two tensors `lhs` and `rhs` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#subtract

    Example:
    ```mlir
    %result = stablehlo.subtract %lhs, %rhs : tensor<2xi32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StableHLO binary logical elementwise op definitions.
//===----------------------------------------------------------------------===//

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class StableHLO_BinaryBiwiseOrLogicalElementwiseOp<string mnemonic> :
        StableHLO_BinaryElementwiseOp<mnemonic,
          [HLO_Commutative, HLO_CompatibleOperandsAndResultType,
           HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect]> {
  let arguments = (ins
    HLO_PredOrIntTensor:$lhs,
    HLO_PredOrIntTensor:$rhs
  );
}

def StableHLO_AndOp: StableHLO_BinaryBiwiseOrLogicalElementwiseOp<"and"> {
  let summary = "And operation";
  let description = [{
    Performs element-wise AND of two tensors `lhs` and `rhs` and produces a
    `result` tensor

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#and

    Example:
    ```mlir
    %result = stablehlo.and %lhs, %rhs : tensor<2x2xi32>
    ```
  }];
}

def StableHLO_OrOp: StableHLO_BinaryBiwiseOrLogicalElementwiseOp<"or"> {
  let summary = "Or operation";
  let description = [{
    Performs element-wise OR of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#or

    Example:
    ```mlir
    %result = stablehlo.or %lhs, %rhs : tensor<2xi1>
    ```
  }];
}

def StableHLO_XorOp : StableHLO_BinaryBiwiseOrLogicalElementwiseOp<"xor"> {
  let summary = "Xor operation";
  let description = [{
    Performs element-wise XOR of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#xor

    Example:
    ```mlir
    %result = stablehlo.xor %lhs, %rhs : tensor<2xi32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// StableHLO control flow op definitions.
//===----------------------------------------------------------------------===//

// Zkx Client API has two separate calls for indexed and predicated conditional,
// although both eventually map to kConditional HLO. IfOp maps to predicated
// conditional use of kConditional HLO.
def StableHLO_IfOp: StableHLO_Op<"if", [
    RecursiveMemoryEffects,
    RecursivelySpeculatable,
    SingleBlockImplicitTerminator<"ReturnOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "If operation";
  let description = [{
    Produces the output from executing exactly one branch from `true_branch` or
    `false_branch` depending on the value of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#if

    Example:
    %result = "stablehlo.if"(%pred) ({
      "stablehlo.return"(%result_true_branch) : (tensor<i32>) -> ()
    }, {
      "stablehlo.return"(%result_false_branch) : (tensor<i32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
  }];

  let arguments = (ins
    HLO_PredTensor:$pred /*if_i1*/
  );

  let regions = (region SizedRegion<1>:$true_branch /*if_i2*/,
                        SizedRegion<1>:$false_branch /*if_i3*/);

  let results = (outs Variadic<HLO_TensorOrToken>);
}

// Zkx Client API has two separate calls for indexed and predicated conditional,
// although both eventually map to kConditional HLO. CaseOp maps to indexed
// conditional use of kConditional HLO.
def StableHLO_CaseOp: StableHLO_Op<"case", [
      RecursiveMemoryEffects,
      RecursivelySpeculatable,
      SingleBlockImplicitTerminator<"ReturnOp">,
      DeclareOpInterfaceMethods<InferTypeOpInterface> /*case_c4*/
    ]> {
  let summary = "Case operation";
  let description = [{
    Produces the output from executing exactly one `function` from `branches`
    depending on the value of `index`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case

    Example:
    ```mlir
    %result0, %result1 = "stablehlo.case"(%index) ({
      stablehlo.return %result_branch0, %result_branch0 : tensor<2xi64>, tensor<2xi64>
    }, {
      stablehlo.return %result_branch1, %result_branch1 : tensor<2xi64>, tensor<2xi64>
    }) : (tensor<i32>) -> (tensor<2xi64>, tensor<2xi64>)
    ```
  }];

  let arguments = (ins
    I32RankedTensor:$index /*case_i1*/
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$branches /*case_i2*/);

  let results = (outs Variadic<HLO_TensorOrToken>);
}


def StableHLO_WhileOp: StableHLO_Op<"while", [
      RecursiveMemoryEffects,
      RecursivelySpeculatable,
      SingleBlockImplicitTerminator<"ReturnOp">,
      DeclareOpInterfaceMethods<InferTypeOpInterface> /*while_c3*/,
      OpAsmOpInterface
    ]> {
  let summary = "While operation";
  let description = [{
    Produces the output from executing `body` function 0 or more times while the
    `cond` function outputs `true`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#while

    Example:
    ```mlir
    %results0, %results1 = stablehlo.while(%arg0 = %init_i, %arg1 = %init_sum) : tensor<i64>, tensor<i64>
    cond {
      %cond = stablehlo.compare LT, %arg0, %ten : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %new_sum = stablehlo.add %arg1, %one : tensor<i64>
      %new_i = stablehlo.add %arg0, %one : tensor<i64>
      stablehlo.return %new_i, %new_sum : tensor<i64>, tensor<i64>
    }
    ```
  }];

  let arguments = (ins Variadic<HLO_TensorOrToken>:$operand /*while_i1*/);

  let regions = (region
    SizedRegion<1>:$cond /*while_i2*/,
    SizedRegion<1>:$body /*while_i3*/
  );

  let results = (outs Variadic<HLO_TensorOrToken>);

  let hasVerifier = 1;

  let extraClassDeclaration = commonClassDeclaration # [{
    // Method of OpAsmOpInterface used during custom printing to name the block
    // arguments in the nested regions. We name both the condition and the body
    // regions entry arguments the same way, with a `iterArg` prefix. Since the
    // two regions are side-by-side they will have the same name, which allows
    // us to print them once and share it for the two regions, and still be able
    // to parse them back.
    void getAsmBlockArgumentNames(Region &region, OpAsmSetValueNameFn setNameFn) {
      for (BlockArgument arg : region.getArguments())
        setNameFn(arg, "iterArg");
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def StableHLO_ReduceOp: StableHLO_ShapedInterfaceOp<"reduce", [
      HLO_RecursivelySpeculatableIfAllInputsStatic,
      RecursiveMemoryEffects,
      SameVariadicOperandSize /*reduce_c3*/,
      InferTensorTypeWithReify /*reduce_c7, reduce_c8*/,
      SingleBlockImplicitTerminator<"ReturnOp">
    ]> { /*reduce_c7*/
  let summary = "Reduce operation";
  let description = [{
    Applies a reduction function `body` to `inputs` and `init_values` along the
    `dimensions` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce

    Example:
    ```mlir
    %result = "stablehlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    ```
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs, /*reduce_i1*/
    Variadic<HLO_Tensor>:$init_values, /*reduce_i2*/
    DenseI64ArrayAttr:$dimensions /*reduce_i3*/
  );
  let regions = (region SizedRegion<1>:$body /*reduce_i4*/);

  // Builder
  // The following custom builder allows inferring the operation type using the
  // 'element_types' of the arguments of the 'body'.
  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$init_values,
      "DenseI64ArrayAttr":$dimensions, "TypeRange":$element_types)>,
  ];

  let results = (outs Variadic<HLO_Tensor>);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// StableHLO tuple op definitions.
//===----------------------------------------------------------------------===//
def StableHLO_GetTupleElementOp: StableHLO_Op<"get_tuple_element", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface> /*get_tuple_element_c2*/]> {
  let summary = "GetTupleElement operation";
  let description = [{
    Extracts element at `index` position of the `operand` tuple and produces a
    `result`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_tuple_element

    Example:
    ```mlir
    %result = stablehlo.get_tuple_element %operand[0] : (tuple<tensor<2xi64>, tuple<tensor<i64>>>) -> tensor<2xi64>
    ```
  }];
  let arguments = (ins
    HLO_Tuple:$operand, /*get_tuple_element_i1*/
    ConfinedAttr<I32Attr, [IntNonNegative]>:$index /*get_tuple_element_c1, get_tuple_element_i2*/
  );

  let results = (outs HLO_TensorOrTokenOrTuple);

  let assemblyFormat = [{
    $operand `[` $index `]` attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_TupleOp : StableHLO_Op<"tuple", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface> /*tuple_c1*/]> {
  let summary = "Tuple operation";
  let description = [{
    Produces a `result` tuple from values `val`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tuple

    Example:
    ```mlir
    %result = stablehlo.tuple %val0, %val1 : tuple<tensor<2xi64>, tuple<tensor<i64>>>
    ```
   }];

  let arguments = (ins Variadic<HLO_TensorOrTokenOrTuple>:$val /*tuple_i1*/);
  let results = (outs HLO_Tuple:$result);

  let assemblyFormat = [{
    $val attr-dict `:` custom<TupleOpType>(type($val), type($result))
  }];
}

def StableHLO_CompareOp: StableHLO_Op<"compare",
    [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
    Elementwise,
    HLO_CompatibleOperandsElementType /*compare_c1*/,
    SameOperandsAndResultShape /*compare_c2*/,
    InferTensorTypeWithReify /*compare_c1, compare_c2*/]> {
  let summary = "Compare operation";
  let description = [{
    Performs element-wise comparison of `lhs` and `rhs` tensors according to
    `comparison_direction` and `compare_type`, and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#compare

    Example:
    ```mlir
    %result = stablehlo.compare LT, %lhs, %rhs : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$lhs /*compare_i1*/,
    HLO_Tensor:$rhs /*compare_i2*/,
    StableHLO_ComparisonDirectionAttr:$comparison_direction /*compare_i3*/
  );
  let results = (outs HLO_PredTensor);

  let assemblyFormat = [{
    $comparison_direction `,` $lhs `,` $rhs
      attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// StableHLO Slice definitions.
//===----------------------------------------------------------------------===//

def StableHLO_SliceOp: StableHLO_Op<
      "slice",
      [HLO_SpeculatableIfStaticDimInOutputIsStaticInInput, NoMemoryEffect,
      SameOperandsAndResultElementType /*slice_c1*/,
      AllMatchSameOperatorTrait<["start_indices", "limit_indices",
      "strides"], "$_self.size()", "size"> /*slice_c2*/,
       DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Slice operation";
  let description = [{
    Extracts a slice from the `operand` using statically-computed starting
    indices and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = stablehlo.slice %operand [1:3, 4:8:2]
       : (tensor<3x8xi64>) -> tensor<2x2xi64>

    // Same in generic form: the `1:3` above is mapped to the first entry in
    // `start_indices` and `limit_indices`, while `strides` is implicitly 1.
    // The `4:8:2` above is parsed into the second entry of `start_indices`,
    // `limit_indices` and `strides` respectively.
    %result = "stablehlo.slice" (%operand) {
      start_indices = array<i64: 1, 4>,
      limit_indices = array<i64: 3, 8>,
      strides = array<i64: 1, 2>
    } : (tensor<3x8xi64>) -> tensor<2x2xi64>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand,
    DenseI64ArrayAttr:$start_indices,
    DenseI64ArrayAttr:$limit_indices,
    DenseI64ArrayAttr:$strides
  );

  let assemblyFormat = [{
    $operand custom<SliceRanges>($start_indices, $limit_indices, $strides)
      attr-dict `:` functional-type(operands, results)
  }];

  let results = (outs HLO_Tensor);
}

def StableHLO_DynamicSliceOp: StableHLO_Op<"dynamic_slice",
      [Pure, AllElementTypesMatch<["operand", "result"]> /*dynamic_slice_c1*/,
       InferTensorType]> {
  let summary = "DynamicSlice operation";
  let description = [{
    Extracts a slice from the `operand` using dynamically-computed starting
    indices and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_slice

    Example:
    ```mlir
    %result = stablehlo.dynamic_slice %operand, %start_indices0, %start_indices1, sizes = [2, 2]
      : (tensor<4x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand /*dynamic_slice_i1*/,
    Variadic<HLO_ScalarIntTensor>:$start_indices /*dynamic_slice_i2*/,
    DenseI64ArrayAttr:$slice_sizes /*dynamic_slice_i3*/
  );

  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $operand `,` custom<VariadicOperandWithAttribute>($start_indices)
      `sizes` `=` $slice_sizes attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_DynamicUpdateSliceOp: StableHLO_Op<"dynamic_update_slice",
      [Pure, AllElementTypesMatch<["operand", "update", "result"]> /*dynamic_update_slice_c1, dynamic_update_slice_c2*/,
       InferTensorType]> {
  let summary = "DynamicUpdateSlice operation";
  let description = [{
    Produces a `result` tensor which is equal to the `operand` tensor except
    that the slice starting at `start_indices` is updated with the values in
    `update`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_update_slice

    Example:
    ```mlir
    %result = stablehlo.dynamic_update_slice %operand, %update, %start_indices0, %start_indices1
      : (tensor<4x4xi32>, tensor<2x2xi32>, tensor<i64>, tensor<i64>) -> tensor<4x4xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand /*dynamic_update_slice_i1*/,
    HLO_Tensor:$update /*dynamic_update_slice_i2*/,
    Variadic<HLO_ScalarIntTensor>:$start_indices /*dynamic_update_slice_i3*/
  );
  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}


//===----------------------------------------------------------------------===//
// StableHLO Other op definitions.
//===----------------------------------------------------------------------===//

def StableHLO_BitcastConvertOp : StableHLO_ShapedInterfaceOp<"bitcast_convert",
    [ConditionallySpeculatable, NoMemoryEffect]> {
  let summary = "BitcastConvert operation";
  let description = [{
    Performs a bitcast operation on `operand` tensor and produces a `result`
    tensor where the bits of the entire `operand` tensor are reinterpreted using
    the type of the `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#bitcast_convert

    Example:
    ```mlir
    %result = stablehlo.bitcast_convert %operand : (tensor<i64>) -> tensor<4xi16>
    ```
  }];

  let arguments = (ins HLO_Tensor:$operand /*bitcast_convert_i1*/);
  let results = (outs HLO_Tensor);
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_BroadcastOp : StableHLO_ShapedInterfaceOp<"broadcast",
    [Pure, SameOperandsAndResultElementType, InferTensorType]> {
  let summary = "Broadcast operation";
  let description = [{
    This operation is on its way out of StableHLO, so it is not included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/3.

    Informally, this operation does the same thing as ZKX's Broadcast:
    https://www.tensorflow.org/xla/operation_semantics#broadcast

    Example:
    ```mlir
    %result = stablehlo.broadcast %operand, sizes = [1, 2] : (tensor<3xi32>) -> tensor<1x2x3xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    DenseI64ArrayAttr:$broadcast_sizes
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $operand `,` `sizes` `=` $broadcast_sizes
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_BroadcastInDimOp : StableHLO_Op<"broadcast_in_dim",
      [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
      HLO_CompatibleOperandsAndResultElementType /*broadcast_in_dim_c1*/]> {
  let summary = "BroadcastInDim operation";
  let description = [{
    Expands the dimensions and/or rank of an input tensor by duplicating the
    data in the `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim

    Example:
    ```mlir
    %result = stablehlo.broadcast_in_dim %operand, dims = [2, 1] : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand /*broadcast_in_dim_i1*/,
    DenseI64ArrayAttr:$broadcast_dimensions /*broadcast_in_dim_i2*/
  );

  let results = (outs HLO_StaticShapeOrBoundedDimTensor);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $operand `,` `dims` `=` $broadcast_dimensions
      attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Determines if this BroadcastInDim instance can be expressed as a Broadcast.
    bool isSimpleBroadcast();
  }];
}

def StableHLO_DynamicBroadcastInDimOp : StableHLO_ShapedInterfaceOp<
    "dynamic_broadcast_in_dim",
    [ConditionallySpeculatable, NoMemoryEffect]> {
  let summary = "DynamicBroadcastInDim operation";
  let description = [{
    This operation is functionally identical to
    [broadcast_in_dim](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#broadcast_in_dim)
    op, but the result shape is specified dynamically via `output_dimensions`.

    It also accepts optional attributes to express static knowledge about the
    expanding behavior of dimensions. If not specified, all dimensions are
    assumed to be possibly expanding. The sets of dimensions that are known to
    be expanding and the set of dimensions that are known to be non-expanding
    must be disjoint and they must be a subset of the operand's dimensions.

    See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_broadcast_in_dim

    Example:
    ```mlir
    %operand = stablehlo.constant dense<[[1, 2, 3]]> : tensor<1x3xi64>
    %output_dimensions = stablehlo.constant dense<[2, 3, 2]> : tensor<3xi64>
    %result = "stablehlo.dynamic_broadcast_in_dim"(%operand, %output_dimensions) {
      broadcast_dimensions = array<i64: 2, 1>,
      known_expanding_dimensions = array<i64: 0>,
      known_nonexpanding_dimensions = array<i64: 1>
    } : (tensor<1x3xi64>, tensor<3xi64>) -> tensor<2x3x2xi64>
    ```

  }];

  let arguments = (ins
    HLO_Tensor:$operand /*dynamic_broadcast_in_dim_i1*/,
    HLO_StaticDimensionTensor:$output_dimensions /*dynamic_broadcast_in_dim_i2*/,
    DenseI64ArrayAttr:$broadcast_dimensions /*dynamic_broadcast_in_dim_i3*/,
    OptionalAttr<DenseI64ArrayAttr>:$known_expanding_dimensions /*dynamic_broadcast_in_dim_i4*/,
    OptionalAttr<DenseI64ArrayAttr>:$known_nonexpanding_dimensions /*dynamic_broadcast_in_dim_i5*/
  );

  let results = (outs HLO_Tensor);

  let builders = [
      OpBuilder<(ins
        "Type":$result_type, "Value":$operand, "Value":$output_dimensions,
        "DenseI64ArrayAttr":$broadcast_dimensions), [{
      build($_builder, $_state, result_type, operand, output_dimensions,
          broadcast_dimensions, /*known_expanding_dimensions=*/{},
          /*known_nonexpanding_dimensions=*/{});
    }]>
  ];

  let hasVerifier = 1;

  let assemblyFormat = [{
    $operand `,` $output_dimensions `,` `dims` `=` $broadcast_dimensions
      attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

// Note: There is no HLO_CallOp because the standard call operation mlir::func::CallOp
// is used instead. A mlir::func::CallOp is exported to a HLO call instruction
// directly.

def StableHLO_ClampOp : StableHLO_ShapedInterfaceOp<"clamp",
    [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
    HLO_CompatibleOperandsAndResultElementType  /* clamp_c3 */, HLO_BroadcastingElementwise,
    InferTensorType]> {
  let summary = "Clamp operation";
  let description = [{
    Clamps every element of the `operand` tensor between a minimum and maximum
    value and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#clamp

    Example:
    ```mlir
    %result = stablehlo.clamp %min, %operand, %max : tensor<3xi32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$min, /*clamp_i1*/
    HLO_Tensor:$operand, /*clamp_c3, clamp_i2*/
    HLO_Tensor:$max /*clamp_i3*/
  );
  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $min `,` $operand `,` $max attr-dict
      `:` custom<SameOperandsAndResultType>(type($min), type($operand), type($max), type($result))
  }];
}

def StableHLO_ConcatenateOp : StableHLO_ShapedInterfaceOp<"concatenate",
    [ConditionallySpeculatable, NoMemoryEffect,
     SameOperandsAndResultElementType /*concatenate_c1, concatenate_c3, concatenate_c5*/,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Concatenate operation";
  let description = [{
    Concatenates a variadic number of tensors in `inputs` along `dimension`
    dimension in the same order as the given arguments and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#concatenate

    Example:
    ```mlir
    %result = stablehlo.concatenate %input0, %input1, dim = 0 : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    ```
  }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs /*concatenate_i1*/,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$dimension /*concatenate_c4, concatenate_i2*/
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
     custom<VariadicOperandWithAttribute>($inputs) `dim` `=` $dimension attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_GetDimensionSizeOp: StableHLO_Op<"get_dimension_size",
      [Pure, InferTensorType]> {
  let summary = "GetDimensionSize operation";
  let description = [{
    Produces the size of the given `dimension` of the `operand`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#get_dimension_size

    Example:
    ```mlir
    %result = stablehlo.get_dimension_size %operand, dim = 1 : (tensor<2x3xi64>) -> tensor<i32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand, /*get_dimension_size_i1*/
    ConfinedAttr<I64Attr, [IntNonNegative]>:$dimension /*get_dimension_size_c1, get_dimension_size_i2*/
  );
  // TODO(hinsu): Allow 64-bit result types once XLA HLO dialect based on the
  // XLA semantics is available. This limitation is because of the current XLA
  // implementation.
  let results = (outs I32RankedTensor);

  let assemblyFormat = [{
    $operand `,` `dim` `=` $dimension attr-dict `:` functional-type(operands, results)
  }];
}


def StableHLO_MapOp: StableHLO_ShapedInterfaceOp<"map",
      [HLO_RecursivelySpeculatableIfAllInputsStatic, RecursiveMemoryEffects,
       SameOperandsAndResultShape /*map_c1, map_c2*/,
       SingleBlockImplicitTerminator<"ReturnOp">, InferTensorTypeWithReify]> {
  let summary = "Map operation";
  let description = [{
    Applies a map function `computation` to `inputs` along the `dimensions` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#map

    Example:
    ```mlir
    %result = "stablehlo.map"(%input0, %input1) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.multiply %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      dimensions = array<i64: 0, 1>
    } : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
    ```
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs /*map_i1*/,
    DenseI64ArrayAttr:$dimensions /*map_i2*/
  );
  let regions = (region SizedRegion<1>:$computation /*map_i3*/);
  let results = (outs HLO_Tensor);

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_ReshapeOp: StableHLO_Op<"reshape",
      [ConditionallySpeculatable, NoMemoryEffect,
       HLO_CompatibleOperandsAndResultElementType]> {
  let summary = "Reshape operation";
  let description = [{
    Performs reshape of `operand` tensor to a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape

    Example:
    ```mlir
    %result = stablehlo.reshape %operand : (tensor<2xi32>) -> tensor<1x2xi32>
    ```
  }];

  let arguments = (ins HLO_Tensor:$operand);

  let results = (outs HLO_StaticShapeOrBoundedDimTensor);
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_DynamicReshapeOp: StableHLO_ShapedInterfaceOp<"dynamic_reshape",
      [HLO_SpeculatableIfAllInputsStaticAndShapeConstant, NoMemoryEffect]> {
  let summary = "DynamicReshape operation";
  let description = [{
    This operation is functionally identical to
    [reshape](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reshape)
    op, but the result shape is specified dynamically via `output_shape`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_reshape

    Example:
    ```mlir
    %output_shape = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %result = stablehlo.dynamic_reshape %operand, %output_shape : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<3x2xi64>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$operand /*dynamic_reshape_i1*/,
    HLO_StaticDimensionTensor:$output_shape /*dynamic_reshape_i2*/
  );
  let results = (outs HLO_Tensor:$result);

  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def StableHLO_ScatterOp: StableHLO_Op<"scatter",
      [ConditionallySpeculatable, RecursiveMemoryEffects,
      SameVariadicOperandSize /*scatter_c5*/,
      DeclareOpInterfaceMethods<InferTypeOpInterface> /*scatter_c24,
      scater_c25*/]> {
  let summary = "Scatter operation";
  let description = [{
    Produces `results` tensors which are equal to `inputs` tensors except that
    several slices specified by `scatter_indices` are updated with the values
    `updates` using `update_computation`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter

   Example:
   ```mlir
   %result = "stablehlo.scatter"(%input, %scatter_indices, %update) ({
     ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
       %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
       stablehlo.return %0 : tensor<i64>
   }) {
     scatter_dimension_numbers = #stablehlo.scatter<
       update_window_dims = [3, 4],
       inserted_window_dims = [1],
       input_batching_dims = [0],
       scatter_indices_batching_dims = [1],
       scatter_dims_to_operand_dims = [2, 1],
       index_vector_dim = 3>,
     indices_are_sorted = false,
     unique_indices = false
   } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
   ```
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs, /*scatter_i1*/
    RankedTensorOf<[AnyInteger, Index]>:$scatter_indices, /*scatter_i2*/
    Variadic<HLO_Tensor>:$updates, /*scatter_i3*/
    StableHLO_ScatterDimensionNumbers:$scatter_dimension_numbers, /*scatter_i4...scatter_i9*/
    DefaultValuedOptionalAttr<BoolAttr, "false">:$indices_are_sorted, /*scatter_i10*/
    DefaultValuedOptionalAttr<BoolAttr, "false">:$unique_indices /*scatter_i11*/
  );

  let regions = (region SizedRegion<1>:$update_computation /*scatter_i12*/);

  let results = (outs Variadic<HLO_Tensor>);

  let hasVerifier = 1;

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}


def StableHLO_SelectOp: StableHLO_Op<"select",
    [HLO_SpeculatableIfAllInputsStatic, NoMemoryEffect,
    HLO_BroadcastingElementwise,
    InferTensorTypeWithReify]> {
  let summary = "Select operation";
  let description = [{
    Produces a `result` tensor where each element is selected from `on_true` or
    `on_false` tensor based on the value of the corresponding element of `pred`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#select

    Example:
    ```mlir
    %result = stablehlo.select %pred, %on_true, %on_false : tensor<2x2xi1>, tensor<2x2xi32>
    ```
  }];
  let arguments = (ins
    HLO_PredTensor:$pred, /*select_i1*/
    HLO_Tensor:$on_true, /*select_i2*/
    HLO_Tensor:$on_false /*select_i3*/
  );

  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    operands attr-dict `:`
      custom<SelectOpType>(type($pred), type($on_true), type($on_false), type($result))
  }];
}

def StableHLO_SetDimensionSizeOp: StableHLO_Op<"set_dimension_size",
      [ConditionallySpeculatable, NoMemoryEffect,
      InferTensorType]> {
  let summary = "SetDimensionSize operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as ZKX's SetDimensionSize:
    https://www.tensorflow.org/xla/operation_semantics#setdimensionsize

    Example:
    ```mlir
    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 1 : (tensor<4x2xi32>, tensor<i32>) -> tensor<4x2xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    I32RankedTensor:$size,
    ConfinedAttr<I64Attr, [IntNonNegative]>:$dimension
  );
  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $operand `,` $size  `,` `dim` `=` $dimension attr-dict
      `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_SortOp : StableHLO_Op<"sort",
      [HLO_RecursivelySpeculatableIfAllInputsStatic, RecursiveMemoryEffects,
      SameOperandsAndResultShape /*sort_c1, sort_c3*/,
      InferTensorType /*sort_c2*/]> {
  let summary = "Sort operation";
  let description = [{
    Sorts a variadic number of tensors in `inputs` together, according to a
    custom `comparator`, along the given `dimension` and produces a variadic
    number of tensors as `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#sort

    Example:
    ```mlir
    %result0, %result1 = "stablehlo.sort"(%input0, %input1) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<i64>):
        %predicate = stablehlo.compare GT, %arg0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %predicate : tensor<i1>
    }) {
      dimension = 0 : i64,
      is_stable = true
    } : (tensor<2x3xi64>, tensor<2x3xi64>) -> (tensor<2x3xi64>, tensor<2x3xi64>)
  }];
  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs /*sort_i1*/,
    DefaultValuedOptionalAttr<I64Attr, "-1">:$dimension /*sort_i2*/,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$is_stable /*sort_i3*/
  );

  let results = (outs Variadic<HLO_Tensor>);

  let regions = (region SizedRegion<1>:$comparator /*sort_i4*/);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, CArg<"int64_t", "-1">:$dimension,
      CArg<"bool", "false">:$is_stable)>];

  let hasVerifier = 1;
}

def StableHLO_ReverseOp: StableHLO_ShapedInterfaceOp<"reverse",
      [HLO_SpeculatableIfStaticDimInOutputIsStaticInInput, NoMemoryEffect,
       SameOperandsAndResultElementType /*reverse_c1*/,
       DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Reverse operation";
  let description = [{
    Reverses the order of elements in the `operand` along the specified
    `dimensions` and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reverse

    Example:
    ```mlir
    %result = stablehlo.reverse %operand, dims = [1] : tensor<3x2xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    DenseI64ArrayAttr:$dimensions
  );

  let hasVerifier = 1;

  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $operand `,` `dims` `=` $dimensions
      attr-dict `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

// TODO(chokobole): Do we need this? Dependency: interior_padding
def StableHLO_PadOp: StableHLO_ShapedInterfaceOp<"pad",
      [HLO_SpeculatableIfStaticDimInOutputIsStaticInInput, NoMemoryEffect,
      SameOperandsAndResultElementType /*pad_c1*/,
      /*pad_c2, pad_i3, pad_i4, pad_i5*/
      AllMatchSameOperatorTrait<["edge_padding_low", "edge_padding_high"], "$_self.size()", "size">,
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Pad operation";
  let description = [{
    Expands `operand` by padding around the tensor as well as between the
    elements of the tensor with the given `padding_value`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad

    Example:
    ```mlir
    %0 = stablehlo.pad %arg0, %arg1, low = [0, 1], high = [2, 1]
      : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x9xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand /*pad_i1*/,
    HLO_ScalarTensor:$padding_value /*pad_i2*/,
    DenseI64ArrayAttr:$edge_padding_low /*pad_i3*/,
    DenseI64ArrayAttr:$edge_padding_high /*pad_i4*/
  );

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $operand `,` $padding_value `,`
      `low` `=` $edge_padding_low `,`
      `high` `=` $edge_padding_high
      attr-dict `:` functional-type(operands, results)
  }];
}

def StableHLO_TransposeOp: StableHLO_ShapedInterfaceOp<"transpose",
      [ConditionallySpeculatable, NoMemoryEffect,
      HLO_CompatibleOperandsAndResultElementType, /*transpose_c1*/
      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Transpose operation";
  let description = [{
    Permutes the dimensions of `operand` tensor using `permutation` and produces
    a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose

    Example:
    ```mlir
    %0 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<1x2x3xi32>) -> tensor<3x2x1xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    DenseI64ArrayAttr:$permutation
  );
  let results = (outs HLO_Tensor:$result);

  let assemblyFormat = [{
    $operand `,` `dims` `=` $permutation
      attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

def StableHLO_ReduceWindowOp: StableHLO_Op<"reduce_window", [
      HLO_RecursivelySpeculatableIfAllInputsStatic,
      RecursiveMemoryEffects,
      SameVariadicOperandSize /*reduce_window_c1*/,
      SingleBlockImplicitTerminator<"ReturnOp">,
      InferTensorType /*reduce_window_c1, reduce_window_c14, reduce_window_c15, reduce_window_c16*/]> {
  let summary = "ReduceWindow operation";
  let description = [{
    Applies a reduction function `body` to windows of `inputs` and `init_values`
    and produces `results`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_window

    Example:
    ```mlir
    %result = "stablehlo.reduce_window"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
        %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
        stablehlo.return %0 : tensor<i64>
    }) {
      window_dimensions = array<i64: 2, 1>,
      window_strides = array<i64: 4, 1>,
      base_dilations = array<i64: 2, 1>,
      window_dilations = array<i64: 3, 1>,
      padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>
    } : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
    ```
  }];

  let arguments = (ins
    Variadic<HLO_Tensor>:$inputs /*reduce_window_i1*/,
    Variadic<HLO_Tensor>:$init_values /*reduce_window_i2*/,
    DenseI64ArrayAttr:$window_dimensions /*reduce_window_i3*/,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the operand dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<DenseI64ArrayAttr>:$window_strides /*reduce_window_i4*/,
    OptionalAttr<DenseI64ArrayAttr>:$base_dilations /*reduce_window_i5*/,
    OptionalAttr<DenseI64ArrayAttr>:$window_dilations /*reduce_window_i6*/,
    OptionalAttr<I64ElementsAttr>:$padding /*reduce_window_i7*/
  );

  let results = (outs Variadic<HLO_Tensor>);

  let regions = (region SizedRegion<1>:$body /*reduce_window_i8*/);

  let hasVerifier = 1;


  // Builder for non-variadic version of the operation.
  let builders = [
    OpBuilder<(ins "Type":$result_type, "Value":$operand,
      "Value":$init_value,
      "DenseI64ArrayAttr":$window_dimensions,
      "DenseI64ArrayAttr":$window_strides,
      "DenseI64ArrayAttr":$base_dilations,
      "DenseI64ArrayAttr":$window_dilations,
      "DenseIntElementsAttr":$padding),
    [{
      build($_builder, $_state, TypeRange(result_type), ValueRange(operand),
            ValueRange(init_value), window_dimensions, window_strides,
            base_dilations, window_dilations, padding);
    }]>,
    OpBuilder<(ins "ValueRange":$operands,
      "ValueRange":$init_values,
      "DenseI64ArrayAttr":$window_dimensions,
      "DenseI64ArrayAttr":$window_strides,
      "DenseI64ArrayAttr":$base_dilations,
      "DenseI64ArrayAttr":$window_dilations,
      "DenseIntElementsAttr":$padding,
      "function_ref<void(OpBuilder &, Location, ValueRange)>":$bodyBuilder
    )>,
  ];
  // TODO(hinsu): Implement custom printer and parser.
}

def StableHLO_ReturnOp : StableHLO_Op<"return", [Pure, Terminator]> {
  let summary = "Return operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/425.

    Informally, this operation serves as a terminator for regions defined by
    the StableHLO ops. Non-StableHLO ops, e.g. `func.func`, have their own
    terminators, e.g. `func.return`.

    Example:
    ```mlir
    %result = "stablehlo.reduce"(%input, %init_value) ({
      ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
        %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
        "stablehlo.return"(%0) : (tensor<i32>) -> ()
    }) {
      dimensions = array<i64: 1>
    } : (tensor<1x6xi32>, tensor<i32>) -> tensor<1xi32>
    ```
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrToken>:$results
  );

  let assemblyFormat = "$results attr-dict (`:` type($results)^)?";
}

def StableHLO_RealDynamicSliceOp: StableHLO_ShapedInterfaceOp<
      "real_dynamic_slice",
      [ConditionallySpeculatable, NoMemoryEffect,
       AllElementTypesMatch<["operand", "result"]>,
       AllTypesMatch<["start_indices", "limit_indices", "strides"]>]> {
  let summary = "RealDynamicSlice operation";
  let description = [{
    This operation is a work in progress, so it is not yet included in
    the StableHLO specification: https://github.com/openxla/stablehlo/issues/8.

    Informally, this operation does the same thing as SliceOp except
    that `start_indices`, `limit_indices` and `strides` are specified dynamically:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = stablehlo.real_dynamic_slice %operand,
                %start_indices, %limit_indices, %strides
           : (tensor<256x?xi32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<256x?xi32>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_DimensionTensor:$start_indices,
    HLO_DimensionTensor:$limit_indices,
    HLO_DimensionTensor:$strides
  );
  let results = (outs HLO_Tensor:$result);
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

// TODO(chokobole): Do we need this? Dependency: interior_padding
def StableHLO_DynamicPadOp: StableHLO_ShapedInterfaceOp<"dynamic_pad",
      [ConditionallySpeculatable, NoMemoryEffect,
      AllElementTypesMatch<["operand", "padding_value", "result"]> /*dynamic_pad_c1*/,
      AllTypesMatch<["edge_padding_low", "edge_padding_high"]> /*dynamic_pad_c2*/,
      AllRanksMatch<["operand", "result"]> /*dynamic_pad_c4*/]> {
  let summary = "DynamicPad operation";
  let description = [{
    This operation is functionally identical to
    [pad](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad)
    https://github.com/openxla/stablehlo/pull/2306#discussion_r1595669709
    op, but with `edge_padding_low`, `edge_padding_high` specified dynamically as values.

    See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dynamic_pad

    Example:
    ```mlir
    %edge_padding_low = stablehlo.constant dense<[0, 1]> : tensor<2xi32>
    %edge_padding_high = stablehlo.constant dense<[2, 1]> : tensor<2xi32>
    %result = stablehlo.dynamic_pad %operand, %padding_value,
                %edge_padding_low, %edge_padding_high
                : (tensor<2x3xi64>, tensor<i64>, tensor<2xi64>, tensor<2xi64>) -> tensor<5x9xi64>
    ```
  }];
  let arguments = (ins
    HLO_Tensor:$operand /*dynamic_pad_i1*/,
    HLO_ScalarTensor:$padding_value /*dynamic_pad_i2*/,
    HLO_StaticDimensionTensor:$edge_padding_low /*dynamic_pad_i3*/,
    HLO_StaticDimensionTensor:$edge_padding_high /*dynamic_pad_i4*/
  );
  let results = (outs HLO_Tensor:$result);
  let hasVerifier = 1;

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = commonClassDeclaration # [{
    /// Interface method for ConditionallySpeculatable.
    mlir::Speculation::Speculatability getSpeculatability();
  }];
}

#endif // ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLO_OPS_TD
