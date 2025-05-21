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
#include "absl/strings/str_cat.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "zkx/base/logging.h"
#include "zkx/layout_util.h"
#include "zkx/math/elliptic_curves/bn/bn254/fr.h"

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

std::string IrName(std::string_view a) {
  std::string s(a);
  s.erase(std::remove(s.begin(), s.end(), '%'), s.end());
  return s;
}

std::string IrName(std::string_view a, std::string_view b) {
  if (!a.empty() && !b.empty()) {
    return IrName(absl::StrCat(a, ".", b));
  }
  return IrName(absl::StrCat(a, b));
}

std::string IrName(const HloInstruction* a, std::string_view b) {
  return IrName(a->name(), b);
}

mlir::Type PrimitiveTypeToMLIRType(PrimitiveType element_type,
                                   mlir::MLIRContext* context) {
  switch (element_type) {
    case S2:
    case U2:
      return mlir::IntegerType::get(context, 2);
    case S4:
    case U4:
      return mlir::IntegerType::get(context, 4);
    case S8:
    case PRED:
    case U8:
      return mlir::IntegerType::get(context, 8);
    case S16:
    case U16:
      return mlir::IntegerType::get(context, 16);
    case S32:
    case U32:
      return mlir::IntegerType::get(context, 32);
    case S64:
    case U64:
      return mlir::IntegerType::get(context, 64);
      // TODO(chokobole): For Tuple, see the comments in
      // ShapeToMLIRMemRefType().
    case TUPLE:
    // An Opaque is like a void*, use i8*.
    case OPAQUE_TYPE:
      return mlir::MemRefType::get({1}, mlir::IntegerType::get(context, 8));
    case TOKEN:
      // Tokens do not have a physical representation, but the compiler needs
      // some placeholder type, so use int8_t*.
      return mlir::MemRefType::get({1}, mlir::IntegerType::get(context, 8));
    case BN254_SCALAR:
      return GetMLIRPrimeFieldType<math::bn254::Fr>(context);
    default:
      LOG(FATAL) << "unsupported type " << element_type;
  }
}

mlir::Type ShapeToMLIRMemRefType(const Shape& shape,
                                 mlir::MLIRContext* context) {
  mlir::Type result_type =
      PrimitiveTypeToMLIRType(shape.element_type(), context);
  if (shape.IsTuple()) {
    // A tuple buffer is an array of pointers.
    // clang-format off
    // TODO(chokobole): Use https://mlir.llvm.org/docs/Dialects/Builtin/#tupletype.
    // clang-format on
    result_type =
        mlir::MemRefType::get({shape.tuple_shapes_size()}, result_type);
  } else if (shape.IsArray()) {
    // TODO(chokobole): Take `major_to_minor` into account.
    std::vector<int64_t> dimensions;
    for (int64_t dimension : shape.dimensions()) {
      dimensions.push_back(dimension);
    }
    result_type = mlir::MemRefType::get(dimensions, result_type);
  }
  return result_type;
}

mlir::Type ShapeToMLIRTensorType(const Shape& shape,
                                 mlir::MLIRContext* context) {
  mlir::Type result_type =
      PrimitiveTypeToMLIRType(shape.element_type(), context);
  if (shape.IsTuple()) {
    // A tuple buffer is an array of pointers.
    // clang-format off
    // TODO(chokobole): Use https://mlir.llvm.org/docs/Dialects/Builtin/#tupletype.
    // clang-format on
    result_type =
        mlir::RankedTensorType::get({shape.tuple_shapes_size()}, result_type);
  } else if (shape.IsArray()) {
    // TODO(chokobole): Take `major_to_minor` into account.
    std::vector<int64_t> dimensions;
    for (int64_t dimension : shape.dimensions()) {
      dimensions.push_back(dimension);
    }
    result_type = mlir::RankedTensorType::get(dimensions, result_type);
  }
  return result_type;
}

}  // namespace zkx::llvm_ir
