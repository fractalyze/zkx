/* Copyright 2022 The StableHLO Authors.
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

#include "llvm/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "zkx/mlir_hlo/stablehlo/dialect/Register.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: mlir::stablehlo::registerXXX
  // mlir::hlo::registerAllTestPasses();
  // mlir::stablehlo::registerPassPipelines();
  // mlir::stablehlo::registerPasses();
  // mlir::stablehlo::registerStablehloLinalgTransformsPasses();
  // mlir::stablehlo::registerInterpreterTransformsPasses();
  // mlir::tosa::registerStablehloTOSATransformsPasses();
  // clang-format on

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::prime_ir::field::FieldDialect>();
  registry.insert<mlir::prime_ir::elliptic_curve::EllipticCurveDialect>();

  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: zkx::stablehlo::check::CheckDialect
  // registry.insert<mlir::stablehlo::check::CheckDialect>();
  // clang-format on
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: zkx::stablehlo::interpreter::InterpreterDialect
  // registry.insert<mlir::stablehlo::interpreter::InterpreterDialect>();
  // clang-format on

  return failed(
      mlir::MlirOptMain(argc, argv, "StableHLO optimizer driver\n", registry));
}
