/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "zkx/mlir_hlo/stablehlo/dialect/Register.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"

#include "zkx/mlir_hlo/stablehlo/dialect/StablehloOps.h"
#include "zkx/mlir_hlo/stablehlo/dialect/VhloOps.h"

namespace mlir::stablehlo {

void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::stablehlo::StablehloDialect,
                  mlir::vhlo::VhloDialect>();
  // clang-format on
}

} // namespace mlir::stablehlo
