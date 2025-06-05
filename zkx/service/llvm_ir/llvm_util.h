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

#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkx/hlo/ir/hlo_instruction.h"
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

// Constructs a human-friendly name from the given inputs.  The result is
// suitable for use as an llvm::Value's name.
//
// This is equivalent to
//
//   - changing the HloInstruction* to its name() (if we called that overload),
//   - joining all of the nonempty inputs by '.', and then
//   - removing all '%'s.
//
std::string IrName(std::string_view a);
std::string IrName(std::string_view a, std::string_view b);
std::string IrName(const HloInstruction* a, std::string_view b = "");

template <typename T>
llvm::APInt ConvertBigIntToAPInt(const T& value) {
  return {T::kBitWidth, T::kLimbNums, value.limbs()};
}

template <typename T>
mlir::zkir::mod_arith::ModArithType GetMLIRModArithType(
    mlir::MLIRContext* context) {
  auto type = mlir::IntegerType::get(context, T::kBitWidth);
  auto modulus =
      mlir::IntegerAttr::get(type, ConvertBigIntToAPInt(T::Config::kModulus));
  return mlir::zkir::mod_arith::ModArithType::get(context, modulus);
}

template <typename T>
mlir::zkir::mod_arith::MontgomeryAttr GetMLIRMontgomeryAttr(
    mlir::MLIRContext* context) {
  return mlir::zkir::mod_arith::MontgomeryAttr::get(
      context, GetMLIRModArithType<T>(context));
}

template <typename T>
mlir::zkir::field::PrimeFieldType GetMLIRPrimeFieldType(
    mlir::MLIRContext* context, bool use_montgomery = T::kUseMontgomery) {
  if constexpr (!T::kUseMontgomery) {
    DCHECK(!use_montgomery);
  }
  auto type = mlir::IntegerType::get(context, T::kBitWidth);
  auto modulus =
      mlir::IntegerAttr::get(type, ConvertBigIntToAPInt(T::Config::kModulus));
  return mlir::zkir::field::PrimeFieldType::get(context, modulus,
                                                use_montgomery);
}

template <typename T>
mlir::zkir::field::PrimeFieldAttr GetMLIRPrimeFieldAttr(
    mlir::MLIRContext* context, const T& value, bool use_montgomery) {
  if constexpr (T::kUseMontgomery) {
    if (use_montgomery) {
      return mlir::zkir::field::PrimeFieldAttr::get(
          GetMLIRPrimeFieldType<T>(context, true),
          ConvertBigIntToAPInt(value.value()));
    } else {
      return mlir::zkir::field::PrimeFieldAttr::get(
          GetMLIRPrimeFieldType<T>(context, false),
          ConvertBigIntToAPInt(value.MontReduce()));
    }
  } else {
    DCHECK(!use_montgomery);
    return mlir::zkir::field::PrimeFieldAttr::get(
        GetMLIRPrimeFieldType<T>(context, false),
        ConvertBigIntToAPInt(value.value()));
  }
}

template <typename T>
mlir::zkir::field::QuadraticExtFieldType GetMLIRQuadraticExtFieldType(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  return mlir::zkir::field::QuadraticExtFieldType::get(
      context, GetMLIRPrimeFieldType<BaseField>(context),
      GetMLIRPrimeFieldAttr(context, T::Config::kNonResidue,
                            BaseField::kUseMontgomery));
}

template <typename T>
mlir::zkir::field::QuadraticExtFieldAttr GetMLIRExtQuadraticExtFieldAttr(
    mlir::MLIRContext* context, const T& value) {
  using BaseField = typename T::BaseField;
  return mlir::zkir::field::QuadraticExtFieldAttr::get(
      context, GetMLIRQuadraticExtFieldType<T>(context),
      GetMLIRPrimeFieldAttr(context, value[0], BaseField::kUseMontgomery),
      GetMLIRPrimeFieldAttr(context, value[1], BaseField::kUseMontgomery));
}

template <typename T>
mlir::zkir::elliptic_curve::ShortWeierstrassAttr GetMLIRG1ShortWeierstrassAttr(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  mlir::zkir::field::PrimeFieldAttr a = GetMLIRPrimeFieldAttr(
      context, T::Curve::Config::kA, BaseField::kUseMontgomery);
  mlir::zkir::field::PrimeFieldAttr b = GetMLIRPrimeFieldAttr(
      context, T::Curve::Config::kB, BaseField::kUseMontgomery);
  mlir::zkir::field::PrimeFieldAttr x = GetMLIRPrimeFieldAttr(
      context, T::Curve::Config::kX, BaseField::kUseMontgomery);
  mlir::zkir::field::PrimeFieldAttr y = GetMLIRPrimeFieldAttr(
      context, T::Curve::Config::kY, BaseField::kUseMontgomery);
  return mlir::zkir::elliptic_curve::ShortWeierstrassAttr::get(context, a, b, x,
                                                               y);
}

template <typename T>
mlir::zkir::elliptic_curve::ShortWeierstrassAttr GetMLIRG2ShortWeierstrassAttr(
    mlir::MLIRContext* context) {
  mlir::zkir::field::QuadraticExtFieldAttr a =
      GetMLIRExtQuadraticExtFieldAttr(context, T::Curve::Config::kA);
  mlir::zkir::field::QuadraticExtFieldAttr b =
      GetMLIRExtQuadraticExtFieldAttr(context, T::Curve::Config::kB);
  mlir::zkir::field::QuadraticExtFieldAttr x =
      GetMLIRExtQuadraticExtFieldAttr(context, T::Curve::Config::kX);
  mlir::zkir::field::QuadraticExtFieldAttr y =
      GetMLIRExtQuadraticExtFieldAttr(context, T::Curve::Config::kY);
  return mlir::zkir::elliptic_curve::ShortWeierstrassAttr::get(context, a, b, x,
                                                               y);
}
template <typename T>
mlir::zkir::elliptic_curve::AffineType GetMLIRAffinePointType(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::zkir::elliptic_curve::AffineType::get(
        context, GetMLIRG1ShortWeierstrassAttr<T>(context));
  } else {
    return mlir::zkir::elliptic_curve::AffineType::get(
        context, GetMLIRG2ShortWeierstrassAttr<T>(context));
  }
}

template <typename T>
mlir::zkir::elliptic_curve::JacobianType GetMLIRJacobianPointType(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::zkir::elliptic_curve::JacobianType::get(
        context, GetMLIRG1ShortWeierstrassAttr<T>(context));
  } else {
    return mlir::zkir::elliptic_curve::JacobianType::get(
        context, GetMLIRG2ShortWeierstrassAttr<T>(context));
  }
}

template <typename T>
mlir::zkir::elliptic_curve::XYZZType GetMLIRPointXyzzType(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::zkir::elliptic_curve::XYZZType::get(
        context, GetMLIRG1ShortWeierstrassAttr<T>(context));
  } else {
    return mlir::zkir::elliptic_curve::XYZZType::get(
        context, GetMLIRG2ShortWeierstrassAttr<T>(context));
  }
}

mlir::Type PrimitiveTypeToMLIRType(PrimitiveType element_type,
                                   mlir::MLIRContext* context);

// Returns the MLIR memref type which represents the given ZKX shape. For
// example, if "shape" is [5 x [10 x i32]], the function returns [5 x 10 x i32].
mlir::Type ShapeToMLIRMemRefType(const Shape& shape,
                                 mlir::MLIRContext* context);

// Returns the MLIR tensor type which represents the given ZKX shape. For
// example, if "shape" is [5 x [10 x i32]], the function returns [5 x 10 x i32].
mlir::Type ShapeToMLIRTensorType(const Shape& shape,
                                 mlir::MLIRContext* context);

}  // namespace zkx::llvm_ir

#endif  // ZKX_SERVICE_LLVM_IR_LLVM_UTIL_H_
