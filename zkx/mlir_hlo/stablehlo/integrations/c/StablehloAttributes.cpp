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

#include "zkx/mlir_hlo/stablehlo/integrations/c/StablehloAttributes.h"

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "zkx/mlir_hlo/stablehlo/dialect/StablehloOps.h"

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

MlirAttribute stablehloComparisonDirectionAttrGet(MlirContext ctx,
                                                  MlirStringRef value) {
  std::optional<mlir::stablehlo::ComparisonDirection> comparisonDirection =
      mlir::stablehlo::symbolizeComparisonDirection(unwrap(value));
  if (!comparisonDirection)
    llvm::report_fatal_error("Invalid value.");
  return wrap(mlir::stablehlo::ComparisonDirectionAttr::get(
      unwrap(ctx), comparisonDirection.value()));
}

bool stablehloAttributeIsAComparisonDirectionAttr(MlirAttribute attr) {
  return llvm::isa<mlir::stablehlo::ComparisonDirectionAttr>(unwrap(attr));
}

MlirStringRef stablehloComparisonDirectionAttrGetValue(MlirAttribute attr) {
  return wrap(mlir::stablehlo::stringifyComparisonDirection(
      llvm::cast<mlir::stablehlo::ComparisonDirectionAttr>(unwrap(attr))
          .getValue()));
}

//===----------------------------------------------------------------------===//
// TypeExtensions
//===----------------------------------------------------------------------===//

MlirAttribute stablehloTypeExtensionsGet(MlirContext ctx, intptr_t nBounds,
                                         const int64_t *bounds) {
  return wrap(mlir::stablehlo::TypeExtensionsAttr::get(
      unwrap(ctx), llvm::ArrayRef(bounds, nBounds)));
}

bool stablehloAttributeIsTypeExtensions(MlirAttribute attr) {
  return llvm::isa<mlir::stablehlo::TypeExtensionsAttr>(unwrap(attr));
}

intptr_t stablehloTypeExtensionsGetBoundsSize(MlirAttribute attr) {
  return llvm::cast<mlir::stablehlo::TypeExtensionsAttr>(unwrap(attr))
      .getBounds()
      .size();
}

int64_t stablehloTypeExtensionsGetBoundsElem(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::stablehlo::TypeExtensionsAttr>(unwrap(attr))
      .getBounds()[pos];
}
