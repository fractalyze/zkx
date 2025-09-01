#ifndef ZKX_MLIR_MLIR_UTILS_H_
#define ZKX_MLIR_MLIR_UTILS_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"

#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "zkir/Dialect/Field/IR/FieldAttributes.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkir/Dialect/Field/IR/FieldTypes.h"
#include "zkir/Dialect/ModArith/IR/ModArithAttributes.h"
#include "zkir/Dialect/ModArith/IR/ModArithTypes.h"
#include "zkx/service/llvm_ir/llvm_util.h"
#include "zkx/shape.h"
#include "zkx/zkx_data.pb.h"

namespace zkx::mlir_utils {

template <typename T>
mlir::zkir::mod_arith::ModArithType GetMlirModArithType(
    mlir::MLIRContext* context) {
  auto type = mlir::IntegerType::get(context, T::kBitWidth);
  auto modulus = mlir::IntegerAttr::get(
      type, llvm_ir::ConvertBigIntToAPInt(T::Config::kModulus));
  return mlir::zkir::mod_arith::ModArithType::get(context, modulus);
}

template <typename T>
mlir::zkir::mod_arith::MontgomeryAttr GetMlirMontgomeryAttr(
    mlir::MLIRContext* context) {
  auto type = mlir::IntegerType::get(context, T::kBitWidth);
  auto modulus = mlir::IntegerAttr::get(
      type, llvm_ir::ConvertBigIntToAPInt(T::Config::kModulus));
  return mlir::zkir::mod_arith::MontgomeryAttr::get(context, modulus);
}

template <typename T>
mlir::zkir::field::PrimeFieldType GetMlirPrimeFieldType(
    mlir::MLIRContext* context, bool use_montgomery) {
  if constexpr (!T::kUseMontgomery) {
    DCHECK(!use_montgomery);
  }
  auto type = mlir::IntegerType::get(context, T::kBitWidth);
  auto modulus = mlir::IntegerAttr::get(
      type, llvm_ir::ConvertBigIntToAPInt(T::Config::kModulus));
  return mlir::zkir::field::PrimeFieldType::get(context, modulus,
                                                use_montgomery);
}

template <typename T>
mlir::zkir::field::PrimeFieldAttr GetMlirPrimeFieldAttr(
    mlir::MLIRContext* context, const T& value, bool use_montgomery) {
  if constexpr (T::kUseMontgomery) {
    if (use_montgomery) {
      return mlir::zkir::field::PrimeFieldAttr::get(
          GetMlirPrimeFieldType<T>(context, true),
          llvm_ir::ConvertBigIntToAPInt(value.value()));
    } else {
      return mlir::zkir::field::PrimeFieldAttr::get(
          GetMlirPrimeFieldType<T>(context, false),
          llvm_ir::ConvertBigIntToAPInt(value.MontReduce().value()));
    }
  } else {
    DCHECK(!use_montgomery);
    return mlir::zkir::field::PrimeFieldAttr::get(
        GetMlirPrimeFieldType<T>(context, false),
        llvm_ir::ConvertBigIntToAPInt(value.value()));
  }
}

template <typename T>
mlir::zkir::field::QuadraticExtFieldType GetMlirQuadraticExtFieldType(
    mlir::MLIRContext* context, bool use_montgomery) {
  using BaseField = typename T::BaseField;
  return mlir::zkir::field::QuadraticExtFieldType::get(
      context, GetMlirPrimeFieldType<BaseField>(context, use_montgomery),
      GetMlirPrimeFieldAttr(context, T::Config::kNonResidue, use_montgomery));
}

template <typename T>
mlir::zkir::field::QuadraticExtFieldAttr GetMlirExtQuadraticExtFieldAttr(
    mlir::MLIRContext* context, const T& value, bool use_montgomery) {
  return mlir::zkir::field::QuadraticExtFieldAttr::get(
      context, GetMlirQuadraticExtFieldType<T>(context, use_montgomery),
      GetMlirPrimeFieldAttr(context, value[0], use_montgomery),
      GetMlirPrimeFieldAttr(context, value[1], use_montgomery));
}

template <typename T>
mlir::zkir::elliptic_curve::ShortWeierstrassAttr GetMLIRG1ShortWeierstrassAttr(
    mlir::MLIRContext* context, bool use_montgomery) {
  mlir::zkir::field::PrimeFieldAttr a =
      GetMlirPrimeFieldAttr(context, T::Curve::Config::kA, use_montgomery);
  mlir::zkir::field::PrimeFieldAttr b =
      GetMlirPrimeFieldAttr(context, T::Curve::Config::kB, use_montgomery);
  mlir::zkir::field::PrimeFieldAttr x =
      GetMlirPrimeFieldAttr(context, T::Curve::Config::kX, use_montgomery);
  mlir::zkir::field::PrimeFieldAttr y =
      GetMlirPrimeFieldAttr(context, T::Curve::Config::kY, use_montgomery);
  return mlir::zkir::elliptic_curve::ShortWeierstrassAttr::get(context, a, b, x,
                                                               y);
}

template <typename T>
mlir::zkir::elliptic_curve::ShortWeierstrassAttr GetMlirG2ShortWeierstrassAttr(
    mlir::MLIRContext* context, bool use_montgomery) {
  mlir::zkir::field::QuadraticExtFieldAttr a = GetMlirExtQuadraticExtFieldAttr(
      context, T::Curve::Config::kA, use_montgomery);
  mlir::zkir::field::QuadraticExtFieldAttr b = GetMlirExtQuadraticExtFieldAttr(
      context, T::Curve::Config::kB, use_montgomery);
  mlir::zkir::field::QuadraticExtFieldAttr x = GetMlirExtQuadraticExtFieldAttr(
      context, T::Curve::Config::kX, use_montgomery);
  mlir::zkir::field::QuadraticExtFieldAttr y = GetMlirExtQuadraticExtFieldAttr(
      context, T::Curve::Config::kY, use_montgomery);
  return mlir::zkir::elliptic_curve::ShortWeierstrassAttr::get(context, a, b, x,
                                                               y);
}
template <typename T>
mlir::zkir::elliptic_curve::AffineType GetMlirAffinePointType(
    mlir::MLIRContext* context, bool use_montgomery) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::zkir::elliptic_curve::AffineType::get(
        context, GetMLIRG1ShortWeierstrassAttr<T>(context, use_montgomery));
  } else {
    return mlir::zkir::elliptic_curve::AffineType::get(
        context, GetMlirG2ShortWeierstrassAttr<T>(context, use_montgomery));
  }
}

template <typename T>
mlir::zkir::elliptic_curve::JacobianType GetMlirJacobianPointType(
    mlir::MLIRContext* context, bool use_montgomery) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::zkir::elliptic_curve::JacobianType::get(
        context, GetMLIRG1ShortWeierstrassAttr<T>(context, use_montgomery));
  } else {
    return mlir::zkir::elliptic_curve::JacobianType::get(
        context, GetMlirG2ShortWeierstrassAttr<T>(context, use_montgomery));
  }
}

template <typename T>
mlir::zkir::elliptic_curve::XYZZType GetMlirPointXyzzType(
    mlir::MLIRContext* context, bool use_montgomery) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::zkir::elliptic_curve::XYZZType::get(
        context, GetMLIRG1ShortWeierstrassAttr<T>(context, use_montgomery));
  } else {
    return mlir::zkir::elliptic_curve::XYZZType::get(
        context, GetMlirG2ShortWeierstrassAttr<T>(context, use_montgomery));
  }
}

// Converts a ZKX primitive type to the corresponding MLIR type.
//
// - Signed/unsigned ZKX primitive types → signless MLIR types
mlir::Type PrimitiveTypeToMlirType(PrimitiveType element_type,
                                   mlir::MLIRContext* context,
                                   bool use_montgomery = false);

// Converts a ZKX primitive type to the corresponding MLIR type,
// preserving signedness where applicable.
//
// - Unsigned ZKX primitive types → unsigned MLIR types
// - Other types → same as PrimitiveTypeToMLIRType()
mlir::Type PrimitiveTypeToMlirTypeWithSign(PrimitiveType element_type,
                                           mlir::MLIRContext* context,
                                           bool use_montgomery = false);

// Returns the MLIR memref type which represents the given ZKX shape. For
// example, if "shape" is [5 x [10 x i32]], the function returns [5 x 10 x i32].
mlir::MemRefType ShapeToMlirMemRefType(const Shape& shape,
                                       mlir::MLIRContext* context);

// Returns the MLIR tensor type which represents the given ZKX shape. For
// example, if "shape" is [5 x [10 x i32]], the function returns [5 x 10 x i32].
mlir::RankedTensorType ShapeToMlirTensorType(const Shape& shape,
                                             mlir::MLIRContext* context);

std::vector<mlir::Type> ShapeToMlirTensorTypes(const Shape& shape,
                                               mlir::MLIRContext* context);

template <typename T>
mlir::Value CreateMlirPrimeFieldConstant(mlir::ImplicitLocOpBuilder& b,
                                         const T& value, bool use_montgomery) {
  if constexpr (T::kUseMontgomery) {
    if (use_montgomery) {
      return b.create<mlir::zkir::field::ConstantOp>(
          GetMlirPrimeFieldType<T>(b.getContext(), true),
          llvm_ir::ConvertBigIntToAPInt(value.value()));
    } else {
      return b.create<mlir::zkir::field::ConstantOp>(
          GetMlirPrimeFieldType<T>(b.getContext(), false),
          llvm_ir::ConvertBigIntToAPInt(value.MontReduce().value()));
    }
  } else {
    DCHECK(!use_montgomery);
    return b.create<mlir::zkir::field::ConstantOp>(
        GetMlirPrimeFieldType<T>(b.getContext(), false),
        llvm_ir::ConvertBigIntToAPInt(value.value()));
  }
}

template <typename T>
mlir::Value CreateMlirQuadraticExtFieldConstant(mlir::ImplicitLocOpBuilder& b,
                                                const T& value,
                                                bool use_montgomery) {
  return b.create<mlir::zkir::field::ConstantOp>(
      GetMlirQuadraticExtFieldType<T>(b.getContext(), use_montgomery),
      GetMlirExtQuadraticExtFieldAttr(b.getContext(), value, use_montgomery));
}

template <typename T>
mlir::Value CreateMlirFieldConstant(mlir::ImplicitLocOpBuilder& b,
                                    const T& value, bool use_montgomery) {
  if constexpr (T::ExtensionDegree() == 1) {
    return CreateMlirPrimeFieldConstant(b, value, use_montgomery);
  } else {
    static_assert(T::ExtensionDegree() == 2);
    return CreateMlirQuadraticExtFieldConstant(b, value, use_montgomery);
  }
}

template <typename T>
mlir::Value CreateMlirEcPointConstant(mlir::ImplicitLocOpBuilder& b,
                                      const T& value, bool use_montgomery) {
  if constexpr (math::IsAffinePoint<T>) {
    llvm::SmallVector<mlir::Value, 2> values;
    values.push_back(CreateMlirFieldConstant(b, value.x(), use_montgomery));
    values.push_back(CreateMlirFieldConstant(b, value.y(), use_montgomery));
    return b.create<mlir::zkir::elliptic_curve::PointOp>(
        GetMlirAffinePointType<T>(b.getContext(), use_montgomery), values);
  } else if constexpr (math::IsJacobianPoint<T>) {
    llvm::SmallVector<mlir::Value, 3> values;
    values.push_back(CreateMlirFieldConstant(b, value.x(), use_montgomery));
    values.push_back(CreateMlirFieldConstant(b, value.y(), use_montgomery));
    values.push_back(CreateMlirFieldConstant(b, value.z(), use_montgomery));
    return b.create<mlir::zkir::elliptic_curve::PointOp>(
        GetMlirJacobianPointType<T>(b.getContext(), use_montgomery), values);
  } else {
    static_assert(math::IsPointXyzz<T>);
    llvm::SmallVector<mlir::Value, 4> values;
    values.push_back(CreateMlirFieldConstant(b, value.x(), use_montgomery));
    values.push_back(CreateMlirFieldConstant(b, value.y(), use_montgomery));
    values.push_back(CreateMlirFieldConstant(b, value.zz(), use_montgomery));
    values.push_back(CreateMlirFieldConstant(b, value.zzz(), use_montgomery));
    return b.create<mlir::zkir::elliptic_curve::PointOp>(
        GetMlirPointXyzzType<T>(b.getContext(), use_montgomery), values);
  }
}

}  // namespace zkx::mlir_utils

#endif  // ZKX_MLIR_MLIR_UTILS_H_
