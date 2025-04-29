/* Copyright 2017 The OpenXLA Authors.

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

#ifndef ZKX_SERVICE_LLVM_IR_LLVM_UTIL_H_
#define ZKX_SERVICE_LLVM_IR_LLVM_UTIL_H_

#include <string>

#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "zkx/shape.h"
#include "zkx/zkx_data.pb.h"

namespace zkx::llvm_ir {

// We have different DumpToString functions for each type for findability. We
// use pointers / values based on the usual semantics of the parameter type.

std::string DumpToString(const llvm::Module* module);
std::string DumpToString(const llvm::Type* type);
std::string DumpToString(const llvm::Value* value);

// This also works for mlir::Op<...> descendants, such as mlir::ModuleOp.
//
// For findability:
//   std::string DumpToString(mlir::Op<...>& op);
//   std::string DumpToString(mlir::ModuleOp& module_op);
//
// The `operation` parameter is not const, because the used print() method is
// not const.
std::string DumpToString(mlir::Operation* operation);
std::string DumpToString(mlir::Type type);
std::string DumpToString(mlir::Value value);

// Returns the LLVM type which represents the given ZKX primitive type.
llvm::Type* PrimitiveTypeToLLVMType(PrimitiveType element_type,
                                    llvm::LLVMContext& context);

// Returns the LLVM type which represents the given ZKX shape. For example,
// if "shape" is [5 x [10 x i32]], the function returns [5 x [10 x i32]].
llvm::Type* ShapeToLLVMType(const Shape& shape, llvm::LLVMContext& context);

// Adds alignment metadata to a load instruction using the given alignment.
// The alignment refers to the result of the load, not the load itself.
void SetAlignmentMetadataForLoad(llvm::LoadInst* load, uint64_t alignment);

// Adds dereferenceable metadata to a load instruction using the given
// the number of dereferenceable bytes.
// Dereferenceable refers to the result of the load, not the load itself.
void SetDereferenceableMetadataForLoad(llvm::LoadInst* load,
                                       uint64_t dereferenceable_bytes);

}  // namespace zkx::llvm_ir

#endif  // ZKX_SERVICE_LLVM_IR_LLVM_UTIL_H_
