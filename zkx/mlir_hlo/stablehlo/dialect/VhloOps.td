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

#ifndef ZKX_MLIR_HLO_STABLEHLO_DIALECT_VHLO_OPS_TD
#define ZKX_MLIR_HLO_STABLEHLO_DIALECT_VHLO_OPS_TD

include "mlir/IR/OpBase.td"
include "zkx/mlir_hlo/stablehlo/dialect/VhloDialect.td"
include "zkx/mlir_hlo/stablehlo/dialect/VhloTypes.td"
include "zkx/mlir_hlo/stablehlo/dialect/VhloEnums.td"
include "zkx/mlir_hlo/stablehlo/dialect/VhloAttrs.td"

def VHLO_VersionedOpInterface : OpInterface<"VersionedOpInterface"> {
  let cppNamespace = "::mlir::vhlo";
  let methods = [
    InterfaceMethod<
      "Returns the minimum version of the VHLO dialect an op is supported in.",
      "mlir::vhlo::Version", "getMinVersion">,
    InterfaceMethod<
      "Returns the maximum version (inclusive) of the VHLO dialect an op is supported in.",
      "mlir::vhlo::Version", "getMaxVersion">,
  ];
}

def VHLO_VersionedOpConstraintInterface : OpInterface<"VersionedOpConstraintInterface"> {
  let cppNamespace = "::mlir::vhlo";
  let methods = [
    InterfaceMethod<
      [{Validate versioned constraints on a versioned op.
       Used if the spec'ed constraints of an op change over time.}],
      "mlir::LogicalResult", "validateConstraint",
      (ins "mlir::Operation*":$op, "mlir::vhlo::Version":$targetVersion)>,
  ];
}

class VHLO_Op<string mnemonic, string minVersion, string maxVersion, list<Trait> traits = []> :
    Op<VHLO_Dialect, mnemonic,
      [DeclareOpInterfaceMethods<VHLO_VersionedOpInterface>] # traits> {
  let extraClassDefinition = [{
    mlir::vhlo::Version $cppClass::getMinVersion() {
      return mlir::vhlo::Version(}] # !subst(".", ", ", minVersion) # [{);
    }
    mlir::vhlo::Version $cppClass::getMaxVersion() {
      }] # !if(
        !eq(maxVersion, "current"),
        [{ return mlir::vhlo::Version::getCurrentVersion(); }],
        [{ return mlir::vhlo::Version(}] # !subst(".", ", ", maxVersion) # [{); }]
      ) # [{
    }
  }];
}

// Design principles for VHLO ops:
// 1) VHLO is a "shallow" versioned copy of the StableHLO dialect that has been
//    simplified down to a bare minimum. As a result, the main design principle
//    is radical minimalism.
// 2) Don't use traits, except when things will break otherwise, i.e.:
//      * ReturnOp needs a trait for Terminator.
//      * ReduceOp/ReduceWindowOp/ScatterOp need a trait since they have
//      * multiple variadic arguments.
// 3) Use VHLO_AnyType or Variadic<VHLO_AnyType> for all operands and results.
// 4) Use VHLO_AnyAttr for all attributes.
// 5) Use VHLO_AnyRegion for all regions.
// 6) Don't use verifiers, shape functions, etc. The only exception is
//    assembly format for FuncOp, because printouts get unreadable otherwise.

def VHLO_AbsOpV1 : VHLO_Op<"abs_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_AddOpV1 : VHLO_Op<"add_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_AndOpV1 : VHLO_Op<"and_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_Atan2OpV1 : VHLO_Op<"atan2_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_BitcastConvertOpV1 : VHLO_Op<"bitcast_convert_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_BroadcastInDimOpV1 : VHLO_Op<"broadcast_in_dim_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$broadcast_dimensions
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#3): BroadcastOp is not part of the StableHLO spec.
// This operation is on its way out of StableHLO, so it is not included in
// the StableHLO specification.
def VHLO_BroadcastOpV1 : VHLO_Op<"broadcast_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$broadcast_sizes
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#425): CallOp is not part of the StableHLO spec.
// We're planning to look into it as part of the work on speccing
// Module/Func/Call/Return ops in StableHLO.
def VHLO_CallOpV1 : VHLO_Op<"call_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyAttr:$callee, Variadic<VHLO_AnyType>:$operands);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_CaseOpV1 : VHLO_Op<"case_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$index);
  let regions = (region VariadicRegion<VHLO_AnyRegion>:$branches);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_ClampOpV1 : VHLO_Op<"clamp_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$min,
    VHLO_AnyType:$operand,
    VHLO_AnyType:$max
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ClzOpV1 : VHLO_Op<"count_leading_zeros_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_CompareOpV1 : VHLO_Op<"compare_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$lhs,
    VHLO_AnyType:$rhs,
    VHLO_AnyAttr:$comparison_direction,
    VHLO_AnyAttr:$compare_type
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ComplexOpV1 : VHLO_Op<"complex_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ConcatenateOpV1 : VHLO_Op<"concatenate_v1", "0.0.1", "current"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyAttr:$dimension
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ConstantOpV1 : VHLO_Op<"constant_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyAttr:$value);
  let results = (outs VHLO_AnyType:$output);
}

def VHLO_ConvertOpV1 : VHLO_Op<"convert_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#3): CreateTokenOp is not part of the StableHLO spec.
// This operation is on its way out of StableHLO, so it is not included in
// the StableHLO specification.
def VHLO_CreateTokenOpV1 : VHLO_Op<"create_token_v1", "0.0.1", "current"> {
  let results = (outs VHLO_AnyType:$output);
}

def VHLO_DivOpV1 : VHLO_Op<"divide_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DynamicBroadcastInDimOpV1 : VHLO_Op<"dynamic_broadcast_in_dim_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$output_dimensions,
    VHLO_AnyAttr:$broadcast_dimensions,
    VHLO_AnyAttr:$known_expanding_dimensions,
    VHLO_AnyAttr:$known_nonexpanding_dimensions
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DynamicIotaOpV1 : VHLO_Op<"dynamic_iota_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$output_shape, VHLO_AnyAttr:$iota_dimension);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DynamicPadOpV1 : VHLO_Op<"dynamic_pad_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$padding_value,
    VHLO_AnyType:$edge_padding_low,
    VHLO_AnyType:$edge_padding_high,
    VHLO_AnyType:$interior_padding
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DynamicReshapeOpV1 : VHLO_Op<"dynamic_reshape_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$operand, VHLO_AnyType:$output_shape);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DynamicSliceOpV1 : VHLO_Op<"dynamic_slice_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    Variadic<VHLO_AnyType>:$start_indices,
    VHLO_AnyAttr:$slice_sizes
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_DynamicUpdateSliceOpV1 : VHLO_Op<"dynamic_update_slice_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$update,
    Variadic<VHLO_AnyType>:$start_indices
  );
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#425): FuncOp is not part of the StableHLO spec.
// We're planning to look into it as part of the work on speccing
// Module/Func/Call/Return ops in StableHLO.
def VHLO_FuncOpV1 : VHLO_Op<"func_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyAttr:$sym_name,
    VHLO_AnyAttr:$function_type,
    VHLO_AnyAttr:$sym_visibility,
    VHLO_AnyAttr:$arg_attrs,
    VHLO_AnyAttr:$res_attrs);
  let regions = (region VHLO_AnyRegion:$body);
  let assemblyFormat = "custom<FunctionBody>($sym_name, $body, $function_type) attr-dict";
}

def VHLO_GetDimensionSizeOpV1 : VHLO_Op<"get_dimension_size_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$dimension
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_GetTupleElementOpV1 : VHLO_Op<"get_tuple_element_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$index
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_IfOpV1 : VHLO_Op<"if_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$pred);
  let regions = (region
    VHLO_AnyRegion:$true_branch,
    VHLO_AnyRegion:$false_branch
  );
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_IotaOpV1 : VHLO_Op<"iota_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyAttr:$iota_dimension);
  let results = (outs VHLO_AnyType:$output);
}

def VHLO_MapOpV1 : VHLO_Op<"map_v1", "0.0.1", "current"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyAttr:$dimensions
  );
  let regions = (region VHLO_AnyRegion:$computation);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_MaxOpV1 : VHLO_Op<"maximum_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_MinOpV1 : VHLO_Op<"minimum_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_MulOpV1 : VHLO_Op<"multiply_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_NegOpV1 : VHLO_Op<"negate_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_NotOpV1 : VHLO_Op<"not_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_OptimizationBarrierOpV1 : VHLO_Op<"optimization_barrier_v1", "0.0.1", "current"> {
  let arguments = (ins Variadic<VHLO_AnyType>:$operand);
  let results = (outs Variadic<VHLO_AnyType>:$result);
}

def VHLO_OrOpV1 : VHLO_Op<"or_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_PadOpV1 : VHLO_Op<"pad_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$padding_value,
    VHLO_AnyAttr:$edge_padding_low,
    VHLO_AnyAttr:$edge_padding_high,
    VHLO_AnyAttr:$interior_padding
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_PopulationCountOpV1 : VHLO_Op<"popcnt_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_PowOpV1 : VHLO_Op<"power_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#8): RealDynamicSliceOp is not part of the StableHLO spec.
// This operation is a work in progress, so it is not yet included in
// the StableHLO specification.
def VHLO_RealDynamicSliceOpV1 : VHLO_Op<"real_dynamic_slice_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$start_indices,
    VHLO_AnyType:$limit_indices,
    VHLO_AnyType:$strides
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ReduceOpV1 : VHLO_Op<"reduce_v1", "0.0.1", "current",
    [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    Variadic<VHLO_AnyType>:$init_values,
    VHLO_AnyAttr:$dimensions
  );
  let regions = (region VHLO_AnyRegion:$body);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_RemOpV1 : VHLO_Op<"remainder_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ReshapeOpV1 : VHLO_Op<"reshape_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

// TODO(#425): ReturnOp is not part of the StableHLO spec.
// We're planning to look into it as part of the work on speccing
// Module/Func/Call/Return ops in StableHLO.
def VHLO_ReturnOpV1 : VHLO_Op<"return_v1", "0.0.1", "current", [Terminator]> {
  let arguments = (ins Variadic<VHLO_AnyType>:$results);
}

def VHLO_ReverseOpV1 : VHLO_Op<"reverse_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$dimensions
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SelectOpV1 : VHLO_Op<"select_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$pred,
    VHLO_AnyType:$on_true,
    VHLO_AnyType:$on_false
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SetDimensionSizeOpV1 : VHLO_Op<"set_dimension_size_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyType:$size,
    VHLO_AnyAttr:$dimension
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ShiftLeftOpV1 : VHLO_Op<"shift_left_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ShiftRightArithmeticOpV1 : VHLO_Op<"shift_right_arithmetic_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_ShiftRightLogicalOpV1 : VHLO_Op<"shift_right_logical_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SignOpV1 : VHLO_Op<"sign_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$operand);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SliceOpV1 : VHLO_Op<"slice_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$start_indices,
    VHLO_AnyAttr:$limit_indices,
    VHLO_AnyAttr:$strides
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_SortOpV1 : VHLO_Op<"sort_v1", "0.0.1", "current"> {
  let arguments = (ins
    Variadic<VHLO_AnyType>:$inputs,
    VHLO_AnyAttr:$dimension,
    VHLO_AnyAttr:$is_stable
  );
  let regions = (region VHLO_AnyRegion:$comparator);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_SubtractOpV1 : VHLO_Op<"subtract_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_TransposeOpV1 : VHLO_Op<"transpose_v1", "0.0.1", "current"> {
  let arguments = (ins
    VHLO_AnyType:$operand,
    VHLO_AnyAttr:$permutation
  );
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_TupleOpV1 : VHLO_Op<"tuple_v1", "0.0.1", "current"> {
  let arguments = (ins Variadic<VHLO_AnyType>:$val);
  let results = (outs VHLO_AnyType:$result);
}

def VHLO_WhileOpV1 : VHLO_Op<"while_v1", "0.0.1", "current"> {
  let arguments = (ins Variadic<VHLO_AnyType>:$operand);
  let regions = (region VHLO_AnyRegion:$cond, VHLO_AnyRegion:$body);
  let results = (outs Variadic<VHLO_AnyType>:$results);
}

def VHLO_XorOpV1 : VHLO_Op<"xor_v1", "0.0.1", "current"> {
  let arguments = (ins VHLO_AnyType:$lhs, VHLO_AnyType:$rhs);
  let results = (outs VHLO_AnyType:$result);
}

#endif // ZKX_MLIR_HLO_STABLEHLO_DIALECT_VHLO_OPS_TD
