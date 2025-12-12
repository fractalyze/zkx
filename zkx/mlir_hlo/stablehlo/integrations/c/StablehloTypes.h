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

#ifndef ZKX_MLIR_HLO_STABLEHLO_INTEGRATIONS_C_STABLEHLOTYPES_H_
#define ZKX_MLIR_HLO_STABLEHLO_INTEGRATIONS_C_STABLEHLOTYPES_H_

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates a token type in the given context.
MLIR_CAPI_EXPORTED MlirType stablehloTokenTypeGet(MlirContext ctx);

// Returns true if the type is a token type.
MLIR_CAPI_EXPORTED bool stablehloTypeIsAToken(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // ZKX_MLIR_HLO_STABLEHLO_INTEGRATIONS_C_STABLEHLOTYPES_H_
