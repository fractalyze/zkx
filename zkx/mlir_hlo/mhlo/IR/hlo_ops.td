/* Copyright 2019 The OpenXLA Authors.

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

// This is the operation definition file for MHLO ops.

#ifndef ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_TD
#define ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_TD

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "zkx/mlir_hlo/mhlo/IR/hlo_utils.td"
include "zkx/mlir_hlo/mhlo/IR/hlo_ops_common.td"

class MHLO_Op<string mnemonic, list<Trait> traits> :
    Op<MHLO_Dialect, mnemonic, traits> {
  // Whether this operation has a custom conversion to HLO or not.
  bit hasCustomHLOConverter = 0b0;

  let extraClassDeclaration = [{
    // Relax the strict default implementation with one that allows
    // for StableHLO-specific differences.
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
}

class MHLO_ShapedInterfaceOp<string mnemonic, list<Trait> traits> :
    MHLO_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferShapedTypeOpInterface,
    ["reifyReturnTypeShapes"]>]> {
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];
}

//===----------------------------------------------------------------------===//
// MHLO nullary op definitions.
//===----------------------------------------------------------------------===//

def MHLO_ConstantOp : MHLO_Op<"constant",
    [ConstantLike, Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constant operation";
  let description = [{
    Produces an `output` tensor from a constant `value`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant

    Example:
    ```mlir
    %output = mhlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    ```
  }];
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    MHLO_StaticShapeTensor:$output
  );

  let builders = [
    OpBuilder<(ins "Attribute":$value)>];

  let hasCustomAssemblyFormat = 1;

  // Constant has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  let hasFolder = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

//===----------------------------------------------------------------------===//
// MHLO unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions

class MHLO_UnaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type OperandType, Type ResultType = OperandType> : MHLO_Op<mnemonic, traits # [Elementwise,
    InferShapedTypeOpInterface, SameOperandsAndResultShape]> {
  let arguments = (ins OperandType:$operand);
  let results = (outs ResultType:$result);
  let extraClassDeclaration = [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                operands.front(),
                                                &reifiedReturnShapes);
    }
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];

  let assemblyFormat = [{
    $operand attr-dict
      `:` custom<SameOperandsAndResultType>(type($operand), type($result))
  }];
}

def MHLO_ConvertOp : MHLO_UnaryElementwiseOp<"convert",
    [Pure, SameOperandsAndResultShape], MHLO_Tensor> {
  let summary = "Convert operation";
  let description = [{
    Performs an element-wise conversion from one element type to another on
    `operand` tensor and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#convert

    Example:
    ```mlir
    %result = mhlo.convert %operand : (tensor<3xi32>) -> tensor<3xcomplex<f32>>
    ```
  }];
  let builders = [
    OpBuilder<(ins "Value":$operand, "Type":$result_element_ty)>];

  let hasFolder = 1;

  let hasCanonicalizer = 1;

  let hasCustomHLOConverter = 1;
}

def MHLO_NegOp: MHLO_UnaryElementwiseOp<"negate",
    [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "Neg operation";
  let description = [{
    Performs element-wise negation of `operand` tensor and produces a `result`
    tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#negate

    Example:
    ```mlir
    %result = mhlo.negate %operand : tensor<2x3xi32>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MHLO binary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations

class MHLO_BinaryElementwiseOp<string mnemonic, list<Trait> traits,
    Type OperandType = MHLO_Tensor, Type ResultType = OperandType> :
    MHLO_Op<mnemonic, traits # [InferShapedTypeOpInterface,
    SameOperandsAndResultShape, Elementwise]> {
  let arguments = (ins
    OperandType:$lhs,
    OperandType:$rhs
  );

  let extraClassDeclaration = [{
    LogicalResult reifyReturnTypeShapes(
        OpBuilder& builder, ValueRange operands,
        SmallVectorImpl<Value>& reifiedReturnShapes) {
      return ::mlir::hlo::deriveShapeFromOperand(&builder, getOperation(),
                                                 operands.front(),
                                                 &reifiedReturnShapes);
    }
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r) {
      return mlir::hlo::isCompatibleForHloTypeInference(l, r);
    }
  }];

  let results = (outs ResultType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
      `:` custom<SameOperandsAndResultType>(type($lhs), type($rhs), type($result))
  }];
}

def MHLO_AddOp : MHLO_BinaryElementwiseOp<"add",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Add operation";
  let description = [{
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#add

    Example:
    ```mlir
    %result = mhlo.add %lhs, %rhs : tensor<2x2xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_DivOp : MHLO_BinaryElementwiseOp<"divide",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "Div operation";
  let description = [{
    Performs element-wise division of dividend `lhs` and divisor `rhs` tensors
    and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#divide

    Example:
    ```mlir
    %result = mhlo.divide %lhs, %rhs : tensor<4xf32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_MulOp : MHLO_BinaryElementwiseOp<"multiply",
      [Commutative, Pure, HLO_CompatibleOperandsAndResultType]> {
  let summary = "Mul operation";
  let description = [{
    Performs element-wise product of two tensors `lhs` and `rhs` and produces a
    `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#multiply

    Example:
    ```mlir
    %result = mhlo.multiply %lhs, %rhs : tensor<2xi32>
    ```
  }];
  let hasFolder = 1;
}

def MHLO_PowOp : MHLO_BinaryElementwiseOp<"power",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "Pow operation";
  let description = [{
    Performs element-wise exponentiation of `lhs` tensor by `rhs` tensor and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#power

    Example:
    ```mlir
    %result = mhlo.power %lhs, %rhs : tensor<6xf32>
    ```
  }];
}

def MHLO_SubtractOp : MHLO_BinaryElementwiseOp<"subtract",
      [Pure, HLO_CompatibleOperandsAndResultType], MHLO_IntTensor> {
  let summary = "Subtract operation";
  let description = [{
    Performs element-wise subtraction of two tensors `lhs` and `rhs` and
    produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#subtract

    Example:
    ```mlir
    %result = mhlo.subtract %lhs, %rhs : tensor<2xi32>
    ```
  }];
  let hasFolder = 1;
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// MHLO tuple op definitions.
//===----------------------------------------------------------------------===//

def MHLO_TupleOp : MHLO_Op<"tuple", [Pure,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Tuple operation";
  let description = [{
    Produces a `result` tuple from values `val`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#tuple

    Example:
    ```mlir
    %result = mhlo.tuple %val0, %val1 : tuple<tensor<2xf32>, tuple<tensor<i32>>>
    ```
   }];
  let arguments = (ins Variadic<MHLO_TensorOrTokenOrTuple>:$val);
  let results = (outs MHLO_Tuple:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $val attr-dict `:` custom<TupleOpType>(type($val), type($result))
  }];
}

//===----------------------------------------------------------------------===//
// MHLO Slice definitions.
//===----------------------------------------------------------------------===//

def MHLO_SliceOp: MHLO_Op<
      "slice",
      [Pure, SameOperandsAndResultElementType,
       AllTypesMatch<["start_indices", "limit_indices", "strides"]>,
       DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Slice operation";
  let description = [{
    Extracts a slice from the `operand` using statically-computed starting
    indices and produces a `result` tensor.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#slice

    Example:
    ```mlir
    %result = "mhlo.slice" (%operand) {
      start_indices = dense<[1, 2]> : tensor<2xi64>,
      limit_indices = dense<[3, 4]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi64>) -> tensor<2x2xi64>
    ```
  }];
  let arguments = (ins
    MHLO_Tensor:$operand,
    I64ElementsAttr:$start_indices,
    I64ElementsAttr:$limit_indices,
    I64ElementsAttr:$strides
  );

  let results = (outs MHLO_Tensor);

  // TODO(chokobole): Uncomment this. Dependency: ConcatenateOp
  // let hasCanonicalizer = 1;
  let hasFolder = 1;
}

#endif // ZKX_MLIR_HLO_MHLO_IR_HLO_OPS_TD
