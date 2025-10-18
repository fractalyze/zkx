#ifndef ZKX_MLIR_MLIR_UTILS_H_
#define ZKX_MLIR_MLIR_UTILS_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
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
#include "zkx/comparison_util.h"
#include "zkx/service/llvm_ir/llvm_util.h"
#include "zkx/shape.h"
#include "zkx/zkx_data.pb.h"

namespace zkx::mlir_utils {

void PopulateTypeConverterWithZkir(mlir::LLVMTypeConverter& converter);

template <size_t N>
llvm::APInt ConvertUnderlyingValueToAPInt(const zk_dtypes::BigInt<N>& value) {
  return llvm_ir::ConvertBigIntToAPInt(value);
}

template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
llvm::APInt ConvertUnderlyingValueToAPInt(T value) {
  return {sizeof(T) * 8, value};
}

template <typename T>
mlir::IntegerAttr GetModulus(mlir::MLIRContext* context) {
  auto type = mlir::IntegerType::get(context, T::kBitWidth);
  return mlir::IntegerAttr::get(
      type, ConvertUnderlyingValueToAPInt(T::Config::kModulus));
}

template <typename T>
mlir::zkir::mod_arith::ModArithType GetMlirModArithType(
    mlir::MLIRContext* context) {
  return mlir::zkir::mod_arith::ModArithType::get(context,
                                                  GetModulus<T>(context));
}

template <typename T>
mlir::zkir::mod_arith::MontgomeryAttr GetMlirMontgomeryAttr(
    mlir::MLIRContext* context) {
  return mlir::zkir::mod_arith::MontgomeryAttr::get(context,
                                                    GetModulus<T>(context));
}

// TODO(chokobole): Remove the `use_montgomery` parameter once
// `zkir::field::RootOfUnityAttr` can be constructed based on the prime field
// representation. Currently, the attribute is always created in standard form,
// regardless of whether the field uses Montgomery form.
template <typename T>
mlir::zkir::field::PrimeFieldType GetMlirPrimeFieldType(
    mlir::MLIRContext* context, bool use_montgomery) {
  if constexpr (!T::kUseMontgomery) {
    DCHECK(!use_montgomery);
  }
  return mlir::zkir::field::PrimeFieldType::get(context, GetModulus<T>(context),
                                                use_montgomery);
}

// TODO(chokobole): Remove `use_montgomery` argument. Refer to the comment
// above.
template <typename T>
mlir::zkir::field::PrimeFieldAttr GetMlirPrimeFieldAttr(
    mlir::MLIRContext* context, const T& value, bool use_montgomery) {
  if constexpr (T::kUseMontgomery) {
    if (use_montgomery) {
      return mlir::zkir::field::PrimeFieldAttr::get(
          GetMlirPrimeFieldType<T>(context, true),
          ConvertUnderlyingValueToAPInt(value.value()));
    } else {
      return mlir::zkir::field::PrimeFieldAttr::get(
          GetMlirPrimeFieldType<T>(context, false),
          ConvertUnderlyingValueToAPInt(value.MontReduce().value()));
    }
  } else {
    DCHECK(!use_montgomery);
    return mlir::zkir::field::PrimeFieldAttr::get(
        GetMlirPrimeFieldType<T>(context, false),
        ConvertUnderlyingValueToAPInt(value.value()));
  }
}

template <typename T>
mlir::zkir::field::QuadraticExtFieldType GetMlirQuadraticExtFieldType(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  return mlir::zkir::field::QuadraticExtFieldType::get(
      context, GetMlirPrimeFieldType<BaseField>(context, T::kUseMontgomery),
      GetMlirPrimeFieldAttr(context, T::Config::kNonResidue,
                            T::kUseMontgomery));
}

template <typename T>
mlir::zkir::field::QuadraticExtFieldAttr GetMlirExtQuadraticExtFieldAttr(
    mlir::MLIRContext* context, const T& value) {
  return mlir::zkir::field::QuadraticExtFieldAttr::get(
      context, GetMlirQuadraticExtFieldType<T>(context),
      GetMlirPrimeFieldAttr(context, value[0], T::kUseMontgomery),
      GetMlirPrimeFieldAttr(context, value[1], T::kUseMontgomery));
}

template <typename T>
mlir::zkir::elliptic_curve::ShortWeierstrassAttr GetMLIRG1ShortWeierstrassAttr(
    mlir::MLIRContext* context) {
  mlir::zkir::field::PrimeFieldAttr a =
      GetMlirPrimeFieldAttr(context, T::Curve::Config::kA, T::kUseMontgomery);
  mlir::zkir::field::PrimeFieldAttr b =
      GetMlirPrimeFieldAttr(context, T::Curve::Config::kB, T::kUseMontgomery);
  mlir::zkir::field::PrimeFieldAttr x =
      GetMlirPrimeFieldAttr(context, T::Curve::Config::kX, T::kUseMontgomery);
  mlir::zkir::field::PrimeFieldAttr y =
      GetMlirPrimeFieldAttr(context, T::Curve::Config::kY, T::kUseMontgomery);
  return mlir::zkir::elliptic_curve::ShortWeierstrassAttr::get(context, a, b, x,
                                                               y);
}

template <typename T>
mlir::zkir::elliptic_curve::ShortWeierstrassAttr GetMlirG2ShortWeierstrassAttr(
    mlir::MLIRContext* context) {
  mlir::zkir::field::QuadraticExtFieldAttr a =
      GetMlirExtQuadraticExtFieldAttr(context, T::Curve::Config::kA);
  mlir::zkir::field::QuadraticExtFieldAttr b =
      GetMlirExtQuadraticExtFieldAttr(context, T::Curve::Config::kB);
  mlir::zkir::field::QuadraticExtFieldAttr x =
      GetMlirExtQuadraticExtFieldAttr(context, T::Curve::Config::kX);
  mlir::zkir::field::QuadraticExtFieldAttr y =
      GetMlirExtQuadraticExtFieldAttr(context, T::Curve::Config::kY);
  return mlir::zkir::elliptic_curve::ShortWeierstrassAttr::get(context, a, b, x,
                                                               y);
}
template <typename T>
mlir::zkir::elliptic_curve::AffineType GetMlirAffinePointType(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::zkir::elliptic_curve::AffineType::get(
        context, GetMLIRG1ShortWeierstrassAttr<T>(context));
  } else {
    return mlir::zkir::elliptic_curve::AffineType::get(
        context, GetMlirG2ShortWeierstrassAttr<T>(context));
  }
}

template <typename T>
mlir::zkir::elliptic_curve::JacobianType GetMlirJacobianPointType(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::zkir::elliptic_curve::JacobianType::get(
        context, GetMLIRG1ShortWeierstrassAttr<T>(context));
  } else {
    return mlir::zkir::elliptic_curve::JacobianType::get(
        context, GetMlirG2ShortWeierstrassAttr<T>(context));
  }
}

template <typename T>
mlir::zkir::elliptic_curve::XYZZType GetMlirPointXyzzType(
    mlir::MLIRContext* context) {
  using BaseField = typename T::BaseField;
  if constexpr (BaseField::ExtensionDegree() == 1) {
    return mlir::zkir::elliptic_curve::XYZZType::get(
        context, GetMLIRG1ShortWeierstrassAttr<T>(context));
  } else {
    return mlir::zkir::elliptic_curve::XYZZType::get(
        context, GetMlirG2ShortWeierstrassAttr<T>(context));
  }
}

// Converts a ZKX primitive type to the corresponding MLIR type.
//
// - Signed/unsigned ZKX primitive types → signless MLIR types
mlir::Type PrimitiveTypeToMlirType(PrimitiveType element_type,
                                   mlir::MLIRContext* context);

// Converts a ZKX primitive type to the corresponding MLIR type,
// preserving signedness where applicable.
//
// - Unsigned ZKX primitive types → unsigned MLIR types
// - Other types → same as PrimitiveTypeToMLIRType()
mlir::Type PrimitiveTypeToMlirTypeWithSign(PrimitiveType element_type,
                                           mlir::MLIRContext* context);

// Returns the MLIR memref type which represents the given ZKX shape. For
// example, if "shape" is [5 x [10 x i32]], the function returns [5 x 10 x i32].
mlir::MemRefType ShapeToMlirMemRefType(const Shape& shape,
                                       mlir::MLIRContext* context);

// Returns the MLIR tensor type which represents the given ZKX shape. For
// example, if "shape" is [5 x [10 x i32]], the function returns [5 x 10 x i32].
mlir::RankedTensorType ShapeToMlirTensorType(const Shape& shape,
                                             mlir::MLIRContext* context);

llvm::SmallVector<mlir::Type> ShapeToMlirTypes(const Shape& shape,
                                               mlir::MLIRContext* context);

template <typename T>
mlir::Value CreateMlirPrimeFieldConstant(mlir::ImplicitLocOpBuilder& b,
                                         const T& value) {
  return b.create<mlir::zkir::field::ConstantOp>(
      GetMlirPrimeFieldType<T>(b.getContext(), T::kUseMontgomery),
      llvm_ir::ConvertBigIntToAPInt(value.value()));
}

template <typename T>
mlir::Value CreateMlirQuadraticExtFieldConstant(mlir::ImplicitLocOpBuilder& b,
                                                const T& value) {
  return b.create<mlir::zkir::field::ConstantOp>(
      GetMlirQuadraticExtFieldType<T>(b.getContext()),
      GetMlirExtQuadraticExtFieldAttr(b.getContext(), value));
}

template <typename T>
mlir::Value CreateMlirFieldConstant(mlir::ImplicitLocOpBuilder& b,
                                    const T& value) {
  if constexpr (T::ExtensionDegree() == 1) {
    return CreateMlirPrimeFieldConstant(b, value);
  } else {
    static_assert(T::ExtensionDegree() == 2);
    return CreateMlirQuadraticExtFieldConstant(b, value);
  }
}

template <typename T>
mlir::Value CreateMlirEcPointConstant(mlir::ImplicitLocOpBuilder& b,
                                      const T& value) {
  if constexpr (zk_dtypes::IsAffinePoint<T>) {
    llvm::SmallVector<mlir::Value, 2> values;
    values.push_back(CreateMlirFieldConstant(b, value.x()));
    values.push_back(CreateMlirFieldConstant(b, value.y()));
    return b.create<mlir::zkir::elliptic_curve::PointOp>(
        GetMlirAffinePointType<T>(b.getContext()), values);
  } else if constexpr (zk_dtypes::IsJacobianPoint<T>) {
    llvm::SmallVector<mlir::Value, 3> values;
    values.push_back(CreateMlirFieldConstant(b, value.x()));
    values.push_back(CreateMlirFieldConstant(b, value.y()));
    values.push_back(CreateMlirFieldConstant(b, value.z()));
    return b.create<mlir::zkir::elliptic_curve::PointOp>(
        GetMlirJacobianPointType<T>(b.getContext()), values);
  } else {
    static_assert(zk_dtypes::IsPointXyzz<T>);
    llvm::SmallVector<mlir::Value, 4> values;
    values.push_back(CreateMlirFieldConstant(b, value.x()));
    values.push_back(CreateMlirFieldConstant(b, value.y()));
    values.push_back(CreateMlirFieldConstant(b, value.zz()));
    values.push_back(CreateMlirFieldConstant(b, value.zzz()));
    return b.create<mlir::zkir::elliptic_curve::PointOp>(
        GetMlirPointXyzzType<T>(b.getContext()), values);
  }
}

mlir::arith::CmpIPredicate CreateMlirArithCmpIPredicate(
    ComparisonDirection direction, bool is_signed);

}  // namespace zkx::mlir_utils

#endif  // ZKX_MLIR_MLIR_UTILS_H_
