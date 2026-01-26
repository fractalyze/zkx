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

  auto scatteredDimsToOperandDimsFunc = [](MlirAttribute self) {
    return attributePropertyVector(
        self, stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize,
        stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem);
  };

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "ScatterDimensionNumbers",
      stablehloAttributeIsAScatterDimensionNumbers)
      .def_classmethod(
          "get",
          [](nb::object cls, const std::vector<int64_t> &updateWindowDims,
             const std::vector<int64_t> &insertedWindowDims,
             const std::vector<int64_t> &inputBatchingDims,
             const std::vector<int64_t> &scatterIndicesBatchingDims,
             const std::vector<int64_t> &scatteredDimsToOperandDims,
             int64_t indexVectorDim, MlirContext ctx) {
            return cls(stablehloScatterDimensionNumbersGet(
                ctx, updateWindowDims.size(), updateWindowDims.data(),
                insertedWindowDims.size(), insertedWindowDims.data(),
                inputBatchingDims.size(), inputBatchingDims.data(),
                scatterIndicesBatchingDims.size(),
                scatterIndicesBatchingDims.data(),
                scatteredDimsToOperandDims.size(),
                scatteredDimsToOperandDims.data(), indexVectorDim));
          },
          nb::arg("cls"), nb::arg("update_window_dims"),
          nb::arg("inserted_window_dims"), nb::arg("input_batching_dims"),
          nb::arg("scatter_indices_batching_dims"),
          nb::arg("scattered_dims_to_operand_dims"),
          nb::arg("index_vector_dim"), nb::arg("context").none() = nb::none(),
          "Creates a ScatterDimensionNumbers with the given dimension "
          "configuration.")
      .def_property_readonly(
          "update_window_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, stablehloScatterDimensionNumbersGetUpdateWindowDimsSize,
                stablehloScatterDimensionNumbersGetUpdateWindowDimsElem);
          })
      .def_property_readonly(
          "inserted_window_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, stablehloScatterDimensionNumbersGetInsertedWindowDimsSize,
                stablehloScatterDimensionNumbersGetInsertedWindowDimsElem);
          })
      .def_property_readonly(
          "input_batching_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self, stablehloScatterDimensionNumbersGetInputBatchingDimsSize,
                stablehloScatterDimensionNumbersGetInputBatchingDimsElem);
          })
      .def_property_readonly(
          "scatter_indices_batching_dims",
          [](MlirAttribute self) {
            return attributePropertyVector(
                self,
                stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize, // NOLINT
                stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem); // NOLINT
          })
      .def_property_readonly("scattered_dims_to_operand_dims",
                             scatteredDimsToOperandDimsFunc)
      .def_property_readonly("index_vector_dim", [](MlirAttribute self) {
        return stablehloScatterDimensionNumbersGetIndexVectorDim(self);
      });

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
