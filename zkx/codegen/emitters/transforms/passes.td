/* Copyright 2025 The OpenXLA Authors.

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

#ifndef ZKX_CODEGEN_EMITTERS_TRANSFORMS_PASSES_TD
#define ZKX_CODEGEN_EMITTERS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertPureCallOpsPass
    : Pass<"zkx-convert-pure-call-ops", "mlir::func::FuncOp"> {
  let summary = "Converts zkx.pure_call to func.call";
  let description = [{
      We use zkx.pure_call ops for calls to enable CSE and other
      transformations (e.g. LICM). This pass rewrites our custom ops to standard
      ops.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];
  let constructor = "CreateConvertPureCallOpsPass()";
}

def EraseDeadFunctionsPass : Pass<"zkx-erase-dead-functions", "mlir::ModuleOp"> {
  let summary = "Deletes unused functions";
  let description = [{
      Deletes functions that are not called.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];

  let constructor = "CreateEraseDeadFunctionsPass()";
}

def FlattenTensorsPass : Pass<"zkx-flatten-tensors", "mlir::ModuleOp"> {
  let summary = "Flatten tensors.";
  let description = [{
    Linearizes all tensors loads and stores.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::vector::VectorDialect",
    "zkx::ZkxDialect",
    "zkx::gpu::ZkxGpuDialect",
  ];
  let constructor = "CreateFlattenTensorsPass()";
}

#endif  // ZKX_CODEGEN_EMITTERS_TRANSFORMS_PASSES_TD
