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

#ifndef ZKX_MLIR_HLO_STABLEHLO_DIALECT_BASE_TD
#define ZKX_MLIR_HLO_STABLEHLO_DIALECT_BASE_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"
include "prime_ir/Dialect/Field/IR/FieldTypes.td"

//===----------------------------------------------------------------------===//
// Common type definitions.
//===----------------------------------------------------------------------===//

def I32RankedTensor : RankedTensorOf<[I32]>;

def UI32RankedTensor : RankedTensorOf<[UI32]>;

//===----------------------------------------------------------------------===//
// HLO type constraints.
//===----------------------------------------------------------------------===//

// Note: Bounded dynamisms is largely unspecced and this feature needs more
// thought as it is adopted to modern frameworks. The current support is
// designed to allow existing TF programs to be representable in StableHLO and
// is subject to change as a formal design for bounded dynamism is developed.
def HLO_HasSingleBoundedDimensionPred
  : CPred<"mlir::hlo::hasSingleBoundedDimension($_self)">;

def HLO_HasStaticOrSingleBoundedShapePred
  : Or<[HasStaticShapePred, HLO_HasSingleBoundedDimensionPred]>;

//===----------------------------------------------------------------------===//
// HLO type definitions.
//===----------------------------------------------------------------------===//

def HLO_Pred : TypeAlias<I1, "pred (AKA boolean or 1-bit integer)">;

// TODO(hinsu): Use signed integers instead of signless integer which is being
// used for legacy reasons.
def HLO_SInt : SignlessIntOfWidths<[2, 4, 8, 16, 32, 64]>;
def HLO_UInt : UnsignedIntOfWidths<[2, 4, 8, 16, 32, 64]>;
def HLO_Int : AnyTypeOf<[HLO_SInt, HLO_UInt]>;

def HLO_Field : AnyTypeOf<[Field_PrimeFieldType, Field_ExtensionFieldType]>;

// Token type.
def HLO_Token : Type<CPred<"isa<TokenType>($_self)">, "token">;

// Any integer tensor types
def HLO_IntTensor : RankedTensorOf<[HLO_Int]>;

// Any integer tensor type with rank 0 (i.e. representing a single integer).
def HLO_ScalarIntTensor : 0DTensorOf<[HLO_Int]>;

def HLO_IntOrFieldTensor : RankedTensorOf<[HLO_Int, HLO_Field]>;

def HLO_PredTensor : RankedTensorOf<[HLO_Pred]>;

def HLO_Tensor : RankedTensorOf<[HLO_Pred, HLO_Int, HLO_Field]>;

def HLO_ScalarTensor: 0DTensorOf<[HLO_Pred, HLO_Int, HLO_Field]>;

def HLO_Tuple : NestedTupleOf<[HLO_Tensor, HLO_Token]>;

def HLO_TensorOrToken : AnyTypeOf<[HLO_Tensor, HLO_Token]>;

def HLO_TensorOrTokenOrTuple : AnyTypeOf<[HLO_Tensor, HLO_Token, HLO_Tuple]>;

def HLO_DimensionValue : AnyTypeOf<[Index, HLO_Int]>;

// Dynamic representation of a shape vector as a tensor.
def HLO_DimensionTensor : 1DTensorOf<[HLO_DimensionValue]>;

//===----------------------------------------------------------------------===//
// Exceptions for unranked dynamism. These should not be used with StableHLO,
// but may be used with CHLO for now.
// TODO(b/326463552): Remove these when CHLO no longer needs unranked dynamism.
//===----------------------------------------------------------------------===//

def HLO_AnyTensor : TensorOf<[HLO_Pred, HLO_Int, HLO_Field]>;

def HLO_AnyPredTensor : TensorOf<[HLO_Pred]>;

def HLO_AnyPredOrIntTensor : TensorOf<[HLO_Pred, HLO_Int]>;

def HLO_AnyTuple : NestedTupleOf<[HLO_AnyTensor, HLO_Token]>;

//===----------------------------------------------------------------------===//
// HLO combined type definitions.
//===----------------------------------------------------------------------===//

// Any integer or predicate tensor types
def HLO_PredOrIntTensor : RankedTensorOf<[HLO_Pred, HLO_Int]>;

//===----------------------------------------------------------------------===//
// HLO static shape type definitions.
//===----------------------------------------------------------------------===//

// Static representation of a shape vector as a tensor.
def HLO_StaticDimensionTensor : RankedTensorOf<[HLO_DimensionValue], [HasStaticShapePred, HasAnyRankOfPred<[1]>], "statically shaped 1-dimensional tensor">;

// Static representation of a 1D tensor of int.
def HLO_Static1DIntTensor : RankedTensorOf<[HLO_Int], [HasStaticShapePred, HasAnyRankOfPred<[1]>], "statically shaped 1-dimensional integer tensor">;

// Static representation of a 2D tensor of int.
def HLO_Static2DIntTensor : RankedTensorOf<[HLO_Int], [HasStaticShapePred, HasAnyRankOfPred<[2]>], "statically shaped 2-dimensional integer tensor">;

// In general, static shaped tensor constraints should be avoided unless
// it is for a legacy op which is only correct with static shapes.
def HLO_StaticShapeTensor : StaticShapeTensorOf<[HLO_Pred, HLO_Int, HLO_Field]>;

def HLO_StaticShapeTensorOrToken : AnyTypeOf<[HLO_StaticShapeTensor, HLO_Token]>;

def HLO_StaticShapeOrBoundedDimTensor : RankedTensorOf<[HLO_Pred, HLO_Int, HLO_Field],
    [HLO_HasStaticOrSingleBoundedShapePred], "statically shaped or single bounded dimension tensor">;

def HLO_StaticShapeIntTensor : StaticShapeTensorOf<[HLO_Int]>;

def HLO_StaticShapeIntOrFieldTensor : StaticShapeTensorOf<[HLO_Int, HLO_Field]>;

//===----------------------------------------------------------------------===//
// HLO traits
//===----------------------------------------------------------------------===//

class HLO_NativeOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::hlo::OpTrait";
}

// An operation that is essentially element-wise but may implement broadcasting
// semantics.
def HLO_BroadcastingElementwise : HLO_NativeOpTrait<"BroadcastingElementwise">;

// This class adds property that the operation is commutative.
// Upstream IsCommutative has default folders, and StableHLO aims to have no
// default folders or canonicalization.
def HLO_Commutative : HLO_NativeOpTrait<"IsCommutative">;

// Op has operand and result types compatible with each other according to
// the rules implemented in isCompatibleForHloTypeInference, which account for
// special properties dynamism, quantization and sparsity.
def HLO_CompatibleOperandsAndResultType : TraitList<
  // TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
  // support quantization or sparsity.
  [
    InferTypeOpInterface,
    DeclareOpInterfaceMethods<InferShapedTypeOpInterface, ["inferReturnTypeComponents"]>,
    HLO_NativeOpTrait<"CompatibleOperandsAndResultType">
  ]>;

def HLO_CompatibleOperandsAndResultElementType :
  HLO_NativeOpTrait<"CompatibleOperandsAndResultElementType">;

def HLO_CompatibleOperandsElementType :
  HLO_NativeOpTrait<"CompatibleOperandsElementType">;

def HLO_BoundedAttrInterface : AttrInterface<"BoundedAttrInterface"> {
  let cppNamespace = "::mlir::hlo";

  let description = [{
    This interface is used for attributes that carry bounds for dimension sizes
    of an accompanying shaped type, e.g. when the attribute represents a
    RankedTensorType::getEncoding.
    The number of bounds is expected to be the same as the number of dimensions
    in the accompanying shaped type.
    For a static dimension, the corresponding bound is ShapedType::kDynamic.
    For a dynamic dimension, the corresponding bound is either known and is
    a non-negative number or unknown and is ShapedType::kDynamic.
  }];

  let methods = [InterfaceMethod<
    "Get the attribute's bounds",
    "::llvm::ArrayRef<int64_t>", "getBounds"
  >];
}

def HLO_SpeculatableIfStaticDimInOutputIsStaticInInputImplTrait
  : HLO_NativeOpTrait<"SpeculatableIfStaticDimInOutputIsStaticInInputImplTrait">;

// This trait can be used with ops where the result has the same rank as the
// input and each dimension of the input maps to the same dimension in the
// result. In that case, if a dim is static in the output but dynamic in the
// input, the dimension could differ at runtime, leading to undefined behavior.
// If the output dimension is dynamic, there is no expectation, so there
// cannot be a mismatch. If the input dimension is static, the output dimension
// can be inferred from it, so there cannot be a mismatch either.
def HLO_SpeculatableIfStaticDimInOutputIsStaticInInput : TraitList<[
    ConditionallySpeculatable, HLO_SpeculatableIfStaticDimInOutputIsStaticInInputImplTrait]>;

def HLO_SpeculatableIfAllInputsStaticImplTrait
  : HLO_NativeOpTrait<"SpeculatableIfAllInputsStaticImplTrait">;

// This trait is appropriate to use for ops where the op is only speculatable
// if all the inputs are static. This can happen if e.g. the inputs are expected
// to all have the same shape. If all the inputs are static, the verifier can
// validate statically that all the static dimensions are the same. However,
// if there is any dynamic dimension, it could differ from the shape of another
// input at runtime, leading to undefined behavior.
def HLO_SpeculatableIfAllInputsStatic : TraitList<[
    ConditionallySpeculatable, HLO_SpeculatableIfAllInputsStaticImplTrait]>;

def HLO_RecursivelySpeculatableIfAllInputsStaticImplTrait
  : HLO_NativeOpTrait<"RecursivelySpeculatableIfAllInputsStaticImplTrait">;

// This trait is the same as HLO_SpeculatableIfAllInputsStatic, but for ops that
// have regions. If all the inputs are static, such an op is
// RecursivelySpeculatable (the ops in its regions have to be checked for
// speculatability).
def HLO_RecursivelySpeculatableIfAllInputsStatic : TraitList<[
    ConditionallySpeculatable, HLO_RecursivelySpeculatableIfAllInputsStaticImplTrait]>;

def HLO_SpeculatableIfAllInputsStaticAndShapeConstantImplTrait
  : HLO_NativeOpTrait<"SpeculatableIfAllInputsStaticAndShapeConstantImplTrait">;

// This trait is the same as HLO_SpeculatableIfAllInputsStatic, but for ops that
// take a shape as their last operand. Such ops are speculatable if all inputs
// are static and the shape is constant.
def HLO_SpeculatableIfAllInputsStaticAndShapeConstant : TraitList<[
    ConditionallySpeculatable, HLO_SpeculatableIfAllInputsStaticAndShapeConstantImplTrait]>;

#endif // ZKX_MLIR_HLO_STABLEHLO_DIALECT_BASE_TD
