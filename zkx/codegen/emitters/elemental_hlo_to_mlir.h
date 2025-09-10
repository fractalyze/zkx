/* Copyright 2024 The OpenXLA Authors.

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
#ifndef ZKX_CODEGEN_EMITTERS_ELEMENTAL_HLO_TO_MLIR_H_
#define ZKX_CODEGEN_EMITTERS_ELEMENTAL_HLO_TO_MLIR_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "zkx/hlo/analysis/indexing_map.h"

namespace zkx::emitters {

// Creates an `apply_indexing` op for the given map.
llvm::SmallVector<mlir::Value, 3> ApplyIndexing(const IndexingMap& map,
                                                mlir::ValueRange dims,
                                                mlir::ValueRange symbols,
                                                mlir::ImplicitLocOpBuilder& b);

// Checks all the constraints and dimension ranges in the map.
mlir::Value CheckConstraints(const IndexingMap& map, mlir::ValueRange dims,
                             mlir::ValueRange symbols,
                             mlir::ImplicitLocOpBuilder& b);

// Populates `lbs`, `ubs` and `steps` with the loop bounds from `indexing_map`.
void GetLoopBoundsFromIndexingMap(mlir::ImplicitLocOpBuilder& b,
                                  const IndexingMap& indexing_map,
                                  llvm::SmallVectorImpl<mlir::Value>* lbs,
                                  llvm::SmallVectorImpl<mlir::Value>* ubs,
                                  llvm::SmallVectorImpl<mlir::Value>* steps);

}  // namespace zkx::emitters

#endif  // ZKX_CODEGEN_EMITTERS_ELEMENTAL_HLO_TO_MLIR_H_
