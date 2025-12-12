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
#ifndef ZKX_MLIR_HLO_STABLEHLO_INTEGRATIONS_C_STABLEHLOATTRIBUTES_H_
#define ZKX_MLIR_HLO_STABLEHLO_INTEGRATIONS_C_STABLEHLOATTRIBUTES_H_

#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute
stablehloComparisonDirectionAttrGet(MlirContext ctx, MlirStringRef value);

MLIR_CAPI_EXPORTED bool
stablehloAttributeIsAComparisonDirectionAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirStringRef
stablehloComparisonDirectionAttrGetValue(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// TypeExtensions
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirAttribute stablehloTypeExtensionsGet(
    MlirContext ctx, intptr_t nBounds, const int64_t *bounds);

MLIR_CAPI_EXPORTED bool stablehloAttributeIsTypeExtensions(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
stablehloTypeExtensionsGetBoundsSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t
stablehloTypeExtensionsGetBoundsElem(MlirAttribute attr, intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif // ZKX_MLIR_HLO_STABLEHLO_INTEGRATIONS_C_STABLEHLOATTRIBUTES_H_
