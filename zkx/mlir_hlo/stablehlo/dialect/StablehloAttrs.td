/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

#ifndef ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLO_ATTRS_TD
#define ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLO_ATTRS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/TensorEncoding.td"
include "mlir/IR/AttrTypeBase.td"

def StableHLO_Dims : ArrayRefParameter<"int64_t", "Dimension"> {
  let parser = "parseDimSizes($_parser)";
  let printer = "printDimSizes($_printer, $_self)";
}

def StableHLO_TypeExtensions : AttrDef<StableHLO_Dialect, "TypeExtensions", [
    DeclareAttrInterfaceMethods<VerifiableTensorEncoding>,
    DeclareAttrInterfaceMethods<HLO_BoundedAttrInterface>]> {
  let mnemonic = "type_extensions";

  // TODO(b/238903065): Move sparsity related info here from the standalone
  // attribute. That will allow composition of bounds and sparsity info.
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$bounds
  );

  let summary = "Attribute that extends tensor type with StableHLO type properties.";

  let description = [{
    This attribute is used to extend MLIR tensor type with StableHLO tensor
    specific properties. These properties aren't modeled in the MLIR type. This
    attribute is set in the `encoding` field of the tensor type.

    See `HLO_BoundedAttrInterface` for documentation for `bounds`.
  }];
  let assemblyFormat = "`<` `bounds` `=` ` ` custom<DimSizes>($bounds) `>`";
}

// A layout attribute (1D tensor of index type)
def StableHLO_LayoutAttr : Attr<
  And<[IndexElementsAttr.predicate,
       CPred<[{cast<::mlir::DenseIntElementsAttr>($_self).getType().getRank()
               == 1}]>]>,
  "A 1D tensor of index type (layout)"> {
  let storageType = IndexElementsAttr.storageType;
  let returnType = IndexElementsAttr.returnType;
  let convertFromStorage = IndexElementsAttr.convertFromStorage;
}

// An array of layout (1D tensor) attributes.
def StableHLO_ArrayOfLayoutAttr : TypedArrayAttrBase<StableHLO_LayoutAttr,
    "Array of layout (1D tensor of index type) attributes">;

// An array of FlatSymbolRef attributes that can be used as a default valued
// attribute.
def StableHLO_FlatSymbolRefArrayAttr :
  TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = "::mlir::ArrayAttr::get($_builder.getContext(), $0)";
}

#endif // ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLO_ATTRS_TD
