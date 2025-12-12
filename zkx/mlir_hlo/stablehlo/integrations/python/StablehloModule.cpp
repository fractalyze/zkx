/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.
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

#include <vector>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#include "zkx/mlir_hlo/stablehlo/integrations/c/StablehloAttributes.h"
#include "zkx/mlir_hlo/stablehlo/integrations/c/StablehloDialect.h"
#include "zkx/mlir_hlo/stablehlo/integrations/c/StablehloTypes.h"

namespace nb = nanobind;

namespace {

// Returns a vector containing integers extracted from an attribute using the
// two provided callbacks.
std::vector<int64_t> attributePropertyVector(
    MlirAttribute attr, llvm::function_ref<intptr_t(MlirAttribute)> sizeFn,
    llvm::function_ref<int64_t(MlirAttribute, intptr_t)> getFn) {
  std::vector<int64_t> result;
  intptr_t size = sizeFn(attr);
  result.reserve(size);
  for (intptr_t i = 0; i < size; ++i) {
    result.push_back(getFn(attr, i));
  }
  return result;
}

auto toPyString(MlirStringRef mlirStringRef) {
  return nb::str(mlirStringRef.data, mlirStringRef.length);
}

} // namespace

NB_MODULE(_stablehlo, m) {
  m.doc() = "stablehlo main python extension";

  //
  // Dialects.
  //

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__stablehlo__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  //
  // Types.
  //

  mlir::python::nanobind_adaptors::mlir_type_subclass(m, "TokenType",
                                                      stablehloTypeIsAToken)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctx) {
            return cls(stablehloTokenTypeGet(ctx));
          },
          nb::arg("cls"), nb::arg("context").none() = nb::none(),
          "Creates a Token type.");

  //
  // Attributes.
  //

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "ComparisonDirectionAttr",
      stablehloAttributeIsAComparisonDirectionAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::string &value, MlirContext ctx) {
            return cls(stablehloComparisonDirectionAttrGet(
                ctx, mlirStringRefCreate(value.c_str(), value.size())));
          },
          nb::arg("cls"), nb::arg("value"),
          nb::arg("context").none() = nb::none(),
          "Creates a ComparisonDirection attribute with the given value.")
      .def_property_readonly("value", [](MlirAttribute self) {
        return toPyString(stablehloComparisonDirectionAttrGetValue(self));
      });

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "TypeExtensions", stablehloAttributeIsTypeExtensions)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<int64_t> &bounds,
             MlirContext ctx) {
            return cls(
                stablehloTypeExtensionsGet(ctx, bounds.size(), bounds.data()));
          },
          nb::arg("cls"), nb::arg("bounds"),
          nb::arg("context").none() = nb::none(),
          "Creates a TypeExtensions with the given bounds.")
      .def_property_readonly("bounds", [](MlirAttribute self) {
        return attributePropertyVector(self,
                                       stablehloTypeExtensionsGetBoundsSize,
                                       stablehloTypeExtensionsGetBoundsElem);
      });
}
