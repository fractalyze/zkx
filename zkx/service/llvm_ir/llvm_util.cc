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

#include "zkx/service/llvm_ir/llvm_util.h"

#include "absl/log/check.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/raw_ostream.h"

#include "zkx/base/logging.h"
#include "zkx/layout_util.h"

namespace zkx::llvm_ir {
namespace {

// This works for most llvm / mlir types. This also accepts a const pointer to
// objects which have a const print() method.
template <typename T>
std::string DumpToStringTempl(T* entity) {
  CHECK_NE(entity, nullptr);

  std::string s;
  llvm::raw_string_ostream ostream(s);
  ostream << *entity;
  return s;
}

}  // namespace

std::string DumpToString(const llvm::Module* module) {
  return DumpToStringTempl(module);
}

std::string DumpToString(const llvm::Type* type) {
  return DumpToStringTempl(type);
}

std::string DumpToString(const llvm::Value* value) {
  return DumpToStringTempl(value);
}

std::string DumpToString(mlir::Operation* operation) {
  return DumpToStringTempl(operation);
}

std::string DumpToString(mlir::Type type) { return DumpToStringTempl(&type); }

std::string DumpToString(mlir::Value value) {
  return DumpToStringTempl(&value);
}

llvm::Type* PrimitiveTypeToLLVMType(PrimitiveType element_type,
                                    llvm::LLVMContext& context) {
  switch (element_type) {
    case S2:
    case U2:
      return llvm::Type::getIntNTy(context, 2);
    case S4:
    case U4:
      return llvm::Type::getIntNTy(context, 4);
    case PRED:
    case S8:
    case U8:
      return llvm::Type::getInt8Ty(context);
    case S16:
    case U16:
      return llvm::Type::getInt16Ty(context);
    case S32:
    case U32:
      return llvm::Type::getInt32Ty(context);
    case S64:
    case U64:
      return llvm::Type::getInt64Ty(context);
    case TUPLE:
    // An Opaque is like a void*, use i8*.
    case OPAQUE_TYPE:
      return llvm::PointerType::getUnqual(context);
    case TOKEN:
      // Tokens do not have a physical representation, but the compiler needs
      // some placeholder type, so use int8_t*.
      return llvm::PointerType::getUnqual(context);
    default:
      LOG(FATAL) << "unsupported type " << element_type;
  }
}

llvm::Type* ShapeToLLVMType(const Shape& shape, llvm::LLVMContext& context) {
  llvm::Type* result_type =
      PrimitiveTypeToLLVMType(shape.element_type(), context);
  if (shape.IsTuple()) {
    // A tuple buffer is an array of pointers.
    result_type = llvm::ArrayType::get(result_type, shape.tuple_shapes_size());
  } else if (shape.IsArray()) {
    for (int64_t dimension : LayoutUtil::MinorToMajor(shape)) {
      result_type =
          llvm::ArrayType::get(result_type, shape.dimensions(dimension));
    }
  }
  return result_type;
}

}  // namespace zkx::llvm_ir
