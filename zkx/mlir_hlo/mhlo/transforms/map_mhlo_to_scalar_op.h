/* Copyright 2019 The OpenXLA Authors.

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

#ifndef ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H_
#define ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H_

#include <type_traits>

#include "absl/log/check.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

#include "zkir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "zkir/Dialect/Field/IR/FieldOps.h"
#include "zkx/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "zkx/service/llvm_ir/llvm_util.h"

namespace mlir::mhlo {
namespace impl {

// A struct to map MhloBinaryOpTy type to the corresponding integer scalar
// operation types.
template <typename MhloBinaryOpTy>
struct MhloToScalarOp {
  using IOp = void;
  using UOp = void;
  using FOp = void;
  using ECOp = void;
};

template <>
struct MhloToScalarOp<mhlo::AddOp> {
  using IOp = arith::AddIOp;
  using UOp = arith::AddIOp;
  using FOp = zkir::field::AddOp;
  using ECOp = zkir::elliptic_curve::AddOp;
};
template <>
struct MhloToScalarOp<mhlo::MulOp> {
  using IOp = arith::MulIOp;
  using UOp = arith::MulIOp;
  using FOp = zkir::field::MulOp;
};
template <>
struct MhloToScalarOp<mhlo::SubtractOp> {
  using IOp = arith::SubIOp;
  using UOp = arith::SubIOp;
  using FOp = zkir::field::SubOp;
  using ECOp = zkir::elliptic_curve::SubOp;
};

// Alias for the map from MHLO binary op type to STD signed integer op type.
template <typename MhloOp>
using ScalarIOp = typename MhloToScalarOp<MhloOp>::IOp;
// Alias for the map from MHLO binary op type to STD unsigned integer op type.
template <typename MhloOp>
using ScalarUOp = typename MhloToScalarOp<MhloOp>::UOp;
// Alias for the map from MHLO binary op type to STD field op type.
template <typename MhloOp>
using ScalarFOp = typename MhloToScalarOp<MhloOp>::FOp;
// Alias for the map from MHLO binary op type to STD elliptic curve op type.
template <typename MhloOp>
using ScalarECOp = typename MhloToScalarOp<MhloOp>::ECOp;

template <typename... Args>
struct MapMhloOpToScalarOpImpl {
  Value operator()(Location /*loc*/, ArrayRef<Type> /*ResultTypes*/,
                   ArrayRef<Type> /*argTypes*/, ValueRange /*args*/,
                   ArrayRef<NamedAttribute> /*attributes*/, OpBuilder* /*b*/) {
    return nullptr;
  }
};

template <typename StdScalarOp>
struct MapMhloOpToScalarOpImpl<StdScalarOp> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> /*argTypes*/, ValueRange args,
                   ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    return b->template create<StdScalarOp>(loc, resultTypes, args, attributes);
  }
};

template <typename SupportedType, typename StdScalarOp, typename... Args>
struct MapMhloOpToScalarOpImpl<SupportedType, StdScalarOp, Args...> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args,
                   ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    Type elementType = getElementTypeOrSelf(argTypes.front());
    if (SupportedType{}(elementType)) {
      return b->template create<StdScalarOp>(loc, resultTypes, args,
                                             attributes);
    }
    return MapMhloOpToScalarOpImpl<Args...>{}(loc, resultTypes, argTypes, args,
                                              attributes, b);
  }
};

template <typename SupportedType, typename... Args>
struct MapMhloOpToScalarOpImpl<SupportedType, void, Args...> {
  Value operator()(Location loc, ArrayRef<Type> resultTypes,
                   ArrayRef<Type> argTypes, ValueRange args,
                   ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    return MapMhloOpToScalarOpImpl<Args...>{}(loc, resultTypes, argTypes, args,
                                              attributes, b);
  }
};

struct IsAnyIntegerType {
  bool operator()(Type t) { return isa<IntegerType>(t); }
};

struct IsSignedIntegerType {
  bool operator()(Type t) {
    // Pretend that signless is signed. This will change eventually.
    return isa<IntegerType>(t) && !t.isUnsignedInteger() &&
           !t.isSignlessInteger(1);
  }
};

struct IsUnsignedIntegerType {
  bool operator()(Type t) {
    return t.isUnsignedInteger() || t.isSignlessInteger(1);
  }
};

struct IsFieldType {
  bool operator()(Type t) {
    return isa<zkir::field::PrimeFieldType>(t) ||
           isa<zkir::field::QuadraticExtFieldType>(t);
  }
};

struct IsEllipticCurveType {
  bool operator()(Type t) {
    return isa<zkir::elliptic_curve::AffineType>(t) ||
           isa<zkir::elliptic_curve::JacobianType>(t) ||
           isa<zkir::elliptic_curve::XYZZType>(t);
  }
};

template <template <typename T> class MapTy, typename OpTy,
          typename PredTy = llvm::is_detected<MapTy, OpTy>>
struct MapableIf {
  using type = void;
};
template <template <typename T> class MapTy, typename OpTy>
struct MapableIf<MapTy, OpTy, std::true_type> {
  using type = MapTy<OpTy>;
};

// Inserts the computation that corresponds to the body of the loop for lowered
// MHLO unary/binary op. Returns the value for the result.
template <typename MhloOpTy>
inline Value mapMhloOpToStdScalarOp(Location loc, ArrayRef<Type> resultTypes,
                                    ArrayRef<Type> argTypes,
                                    typename MhloOpTy::Adaptor adaptor,
                                    ArrayRef<NamedAttribute> attributes,
                                    OpBuilder* b) {
  using ScalarIOpOrVoid = typename MapableIf<ScalarIOp, MhloOpTy>::type;
  using ScalarUOpOrVoid = typename MapableIf<ScalarUOp, MhloOpTy>::type;
  using ScalarFOpOrVoid = typename MapableIf<ScalarFOp, MhloOpTy>::type;
  using ScalarECOpOrVoid = typename MapableIf<ScalarECOp, MhloOpTy>::type;
  return MapMhloOpToScalarOpImpl<IsSignedIntegerType, ScalarIOpOrVoid,
                                 IsUnsignedIntegerType, ScalarUOpOrVoid,
                                 IsFieldType, ScalarFOpOrVoid,
                                 IsEllipticCurveType, ScalarECOpOrVoid>{}(
      loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
}

// Return a constant for v of type t, splat if t is a vector type.
inline Value getConstantOrSplat(OpBuilder* b, Location loc, Type t,
                                Attribute v) {
  if (VectorType vecType = dyn_cast<VectorType>(t)) {
    v = SplatElementsAttr::get(vecType, v);
  }
  return b->create<arith::ConstantOp>(loc, t, cast<TypedAttr>(v));
}

inline Value makeIntConvert(Location loc, ArrayRef<Type> resultTypes,
                            Type sourceType, Type targetType, ValueRange args,
                            ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
  if (targetType.isInteger(/*width=*/1)) {
    // When casting to bool, we need to compare whether the value is equal to
    // zero.
    Value zeroIntval = b->create<arith::ConstantOp>(
        loc, b->getZeroAttr(args.front().getType()));
    return b->create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, args.front(),
                                    zeroIntval);
  }
  auto src = cast<IntegerType>(sourceType);
  auto res = cast<IntegerType>(targetType);
  if (src.getWidth() > res.getWidth()) {
    return b->create<arith::TruncIOp>(loc, resultTypes, args, attributes);
  }
  if (src.getWidth() < res.getWidth()) {
    // Special case boolean values, so they get casted to `1` instead of `-1`.
    if (IsUnsignedIntegerType{}(src)) {
      return b->create<arith::ExtUIOp>(loc, resultTypes, args, attributes);
    }
    return b->create<arith::ExtSIOp>(loc, resultTypes, args, attributes);
  }
  // No conversion is needed for the same width integers
  return args.front();
}

inline Value makeFieldConvert(Location loc, ArrayRef<Type> resultTypes,
                              Type sourceType, Type targetType, ValueRange args,
                              ArrayRef<NamedAttribute> attributes,
                              OpBuilder* b) {
  if (zkir::field::isMontgomery(sourceType)) {
    if (zkir::field::isMontgomery(targetType)) {
      return args[0];
    } else {
      return b->create<zkir::field::FromMontOp>(loc, resultTypes, args,
                                                attributes);
    }
  } else {
    if (zkir::field::isMontgomery(targetType)) {
      return b->create<zkir::field::ToMontOp>(loc, resultTypes, args,
                                              attributes);
    } else {
      return args[0];
    }
  }
}

inline Value makeEcPointConvert(Location loc, ArrayRef<Type> resultTypes,
                                Type sourceType, Type targetType,
                                ValueRange args,
                                ArrayRef<NamedAttribute> attributes,
                                OpBuilder* b) {
  if (isa<zkir::elliptic_curve::AffineType>(sourceType)) {
    if (isa<zkir::elliptic_curve::JacobianType>(targetType) ||
        isa<zkir::elliptic_curve::XYZZType>(targetType)) {
      return b->create<zkir::elliptic_curve::ConvertPointTypeOp>(
          loc, resultTypes, args, attributes);
    }
  } else if (isa<zkir::elliptic_curve::JacobianType>(sourceType)) {
    if (isa<zkir::elliptic_curve::AffineType>(targetType) ||
        isa<zkir::elliptic_curve::XYZZType>(targetType)) {
      return b->create<zkir::elliptic_curve::ConvertPointTypeOp>(
          loc, resultTypes, args, attributes);
    }
  } else if (isa<zkir::elliptic_curve::XYZZType>(sourceType)) {
    if (isa<zkir::elliptic_curve::AffineType>(targetType) ||
        isa<zkir::elliptic_curve::JacobianType>(targetType)) {
      return b->create<zkir::elliptic_curve::ConvertPointTypeOp>(
          loc, resultTypes, args, attributes);
    }
  }
  return args[0];
}

inline Value mapConvertOpToStdScalarOp(Location loc, ArrayRef<Type> targetTypes,
                                       ArrayRef<Type> resultTypes,
                                       ArrayRef<Type> argTypes, ValueRange args,
                                       ArrayRef<NamedAttribute> attributes,
                                       OpBuilder* b) {
  assert(targetTypes.size() == 1 && "ConvertOp should return a single result");
  assert(resultTypes.size() == 1 && "ConvertOp should return a single result");
  assert(argTypes.size() == 1 && "ConvertOp should take a single argument");
  assert(args.size() == 1 && "ConvertOp should take a single argument");

  Type sourceType = getElementTypeOrSelf(argTypes.front());
  Type targetType = getElementTypeOrSelf(targetTypes.front());

  if (isa<IntegerType>(sourceType) && isa<IntegerType>(targetType)) {
    return makeIntConvert(loc, resultTypes, sourceType, targetType, args,
                          attributes, b);
  } else if (isa<zkir::field::PrimeFieldType>(sourceType) ||
             isa<zkir::field::QuadraticExtFieldType>(sourceType)) {
    return makeFieldConvert(loc, resultTypes, sourceType, targetType, args,
                            attributes, b);
  } else if (isa<zkir::elliptic_curve::AffineType>(sourceType) ||
             isa<zkir::elliptic_curve::JacobianType>(sourceType) ||
             isa<zkir::elliptic_curve::XYZZType>(sourceType)) {
    return makeEcPointConvert(loc, resultTypes, sourceType, targetType, args,
                              attributes, b);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::MulOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::MulOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  Type leftType = adaptor.getLhs().getType();
  Type leftElementType = getElementTypeOrSelf(leftType);
  if (IsAnyIntegerType{}(leftElementType)) {
    using ScalarIOpOrVoid = typename MapableIf<ScalarIOp, mhlo::MulOp>::type;
    using ScalarUOpOrVoid = typename MapableIf<ScalarUOp, mhlo::MulOp>::type;
    return MapMhloOpToScalarOpImpl<IsSignedIntegerType, ScalarIOpOrVoid,
                                   IsUnsignedIntegerType, ScalarUOpOrVoid>{}(
        loc, resultTypes, argTypes, adaptor.getOperands(), attributes, b);
  } else if (IsFieldType{}(leftElementType)) {
    Type rightType = adaptor.getRhs().getType();
    Type rightElementType = getElementTypeOrSelf(rightType);
    if (IsEllipticCurveType{}(rightElementType)) {
      return b->create<zkir::elliptic_curve::ScalarMulOp>(
          loc, resultTypes, adaptor.getOperands(), attributes);
    } else {
      return b->create<zkir::field::MulOp>(loc, resultTypes,
                                           adaptor.getOperands(), attributes);
    }
  }
  return nullptr;
}

template <typename U, typename S>
inline Value makeSafeIntDiv(ImplicitLocOpBuilder& lb, Type originalType,
                            Value lhs, Value rhs, Value returnedOnZero,
                            Value returnedOnSignedOverflow) {
  Type type = lhs.getType();
  auto elementType = cast<IntegerType>(getElementTypeOrSelf(type));
  Value zero = lb.create<arith::ConstantOp>(lb.getZeroAttr(type));
  auto makeConstant = [&](const APInt& i) {
    return getConstantOrSplat(&lb, lb.getLoc(), type,
                              lb.getIntegerAttr(elementType, i));
  };
  Value one = makeConstant(APInt(elementType.getWidth(), 1));
  Value rhsIsZero =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, rhs, zero);

  // For unsigned just set the divisor to 1 when it would be 0.
  if (originalType.isUnsignedInteger()) {
    Value safeRhs = lb.create<arith::SelectOp>(rhsIsZero, one, rhs);
    Value safeDiv = lb.create<U>(lhs, safeRhs);
    return lb.create<arith::SelectOp>(rhsIsZero, returnedOnZero, safeDiv);
  }

  // For signed also check for INT_MIN / -1.
  Value smin = makeConstant(APInt::getSignedMinValue(elementType.getWidth()));
  Value lhsIsSmin =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, smin);
  Value minusOne = makeConstant(APInt::getAllOnes(elementType.getWidth()));
  Value rhsIsMinusOne =
      lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq, rhs, minusOne);
  Value hasIntMinOverflow = lb.create<arith::AndIOp>(lhsIsSmin, rhsIsMinusOne);
  Value rhsIsUnsafe = lb.create<arith::OrIOp>(rhsIsZero, hasIntMinOverflow);
  Value safeRhs = lb.create<arith::SelectOp>(rhsIsUnsafe, one, rhs);
  Value safeDiv = lb.create<S>(lhs, safeRhs);
  Value safeSmin = lb.create<arith::SelectOp>(
      hasIntMinOverflow, returnedOnSignedOverflow, safeDiv);
  return lb.create<arith::SelectOp>(rhsIsZero, returnedOnZero, safeSmin);
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::DivOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::DivOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  Type type = adaptor.getLhs().getType();
  Type elementType = getElementTypeOrSelf(type);
  if (IsAnyIntegerType{}(elementType)) {
    // Integer division overflow behavior:
    //
    // X / 0 == -1
    // INT_SMIN /s -1 = INT_SMIN
    ImplicitLocOpBuilder lb(loc, *b);

    auto integerElementType = cast<IntegerType>(elementType);
    auto makeConstant = [&](const APInt& i) {
      return getConstantOrSplat(&lb, lb.getLoc(), type,
                                lb.getIntegerAttr(integerElementType, i));
    };
    Value minusOne =
        makeConstant(APInt::getAllOnes(integerElementType.getWidth()));
    Value smin =
        makeConstant(APInt::getSignedMinValue(integerElementType.getWidth()));
    Type originalType = getElementTypeOrSelf(argTypes.front());
    return makeSafeIntDiv<arith::DivUIOp, arith::DivSIOp>(
        lb, originalType, adaptor.getLhs(), adaptor.getRhs(),
        /*returnedOnZero=*/minusOne,
        /*returnedOnSignedOverflow=*/smin);
  } else if (IsFieldType{}(elementType)) {
    auto inv = b->create<zkir::field::InverseOp>(loc, adaptor.getRhs());
    return b->create<zkir::field::MulOp>(loc, adaptor.getLhs(), inv);
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::NegOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::NegOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  Type elementType = getElementTypeOrSelf(adaptor.getOperand().getType());
  if (IsAnyIntegerType{}(elementType)) {
    // lmhlo.neg(x, result) -> result = sub(0, x)
    Value lhs = adaptor.getOperand();
    Value zeroIntval =
        b->create<arith::ConstantOp>(loc, b->getZeroAttr(lhs.getType()));
    return b->create<ScalarIOp<mhlo::SubtractOp>>(loc, zeroIntval, lhs);
  } else if (IsFieldType{}(elementType)) {
    return b->create<zkir::field::NegateOp>(loc, adaptor.getOperand());
  } else if (IsEllipticCurveType{}(elementType)) {
    return b->create<zkir::elliptic_curve::NegateOp>(loc, adaptor.getOperand());
  }
  return nullptr;
}

template <>
inline Value mapMhloOpToStdScalarOp<mhlo::PowOp>(
    Location loc, ArrayRef<Type> resultTypes, ArrayRef<Type> argTypes,
    mhlo::PowOp::Adaptor adaptor, ArrayRef<NamedAttribute> attributes,
    OpBuilder* b) {
  Type type = adaptor.getLhs().getType();
  Type elementType = getElementTypeOrSelf(type);
  if (IsAnyIntegerType{}(elementType)) {
    // TODO: b/315868720 Consider alternate lowerings of mhlo::PowOp with
    // integer operands.
    auto lb = ImplicitLocOpBuilder(loc, *b);
    auto resultType = getElementTypeOrSelf(resultTypes.front());
    // Exponentiation by squaring:
    // https://en.wikipedia.org/wiki/Exponentiation_by_squaring;
    Value negOne =
        lb.create<arith::ConstantOp>(lb.getIntegerAttr(resultType, -1));
    Value zero = lb.create<arith::ConstantOp>(lb.getIntegerAttr(resultType, 0));
    Value one = lb.create<arith::ConstantOp>(lb.getIntegerAttr(resultType, 1));
    Value two = lb.create<arith::ConstantOp>(lb.getIntegerAttr(resultType, 2));
    Value step = lb.create<arith::ConstantIndexOp>(1);
    Value lowerBound = lb.create<arith::ConstantIndexOp>(0);
    // Everything else would overflow for any exponent > 1, as 2^64
    // is the largest possible exponent for a 64-bit integer, and
    // that's 1 << 6.
    Value upperBound = lb.create<arith::ConstantIndexOp>(6);
    auto originalBase = adaptor.getLhs();
    auto originalExponent = adaptor.getRhs();

    Value accum =
        lb.create<scf::ForOp>(
              lowerBound, upperBound, step,
              SmallVector<Value>({one, originalBase, originalExponent}),
              [&](OpBuilder& b, Location, Value /*v*/, ValueRange iters) {
                Value accum = iters[0];
                Value base = iters[1];
                Value exponent = iters[2];

                Value condition = b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::eq,
                    b.create<arith::AndIOp>(loc, exponent, one), one);
                Value multiplied = b.create<arith::MulIOp>(loc, accum, base);
                accum = b.create<arith::SelectOp>(loc, condition, multiplied,
                                                  accum);
                base = b.create<arith::MulIOp>(loc, base, base);
                exponent = b.create<arith::ShRUIOp>(loc, exponent, one);
                b.create<scf::YieldOp>(
                    loc, SmallVector<Value>({accum, base, exponent}));
              })
            .getResult(0);

    Value rhsIsEven = lb.create<arith::CmpIOp>(
        arith::CmpIPredicate::eq,
        lb.create<arith::RemSIOp>(adaptor.getRhs(), two), zero);
    Value rhsIsNegative = lb.create<arith::CmpIOp>(arith::CmpIPredicate::slt,
                                                   adaptor.getRhs(), zero);
    Value lhsIsOne = lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                              adaptor.getLhs(), one);
    Value lhsIsNegOne = lb.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                 adaptor.getLhs(), negOne);

    // The accum is correct when the rhs is non-negative. When rhs is
    // negative, we return 0 for integer, with the exception of lhs values of 1
    // and -1 which have integer results for negative exponents. Specifically,
    // the calculation is the following:
    //
    // - Return accum if the rhs is not negative.
    // - Return 1 or -1 depending on the parity of rhs when the lhs is -1.
    // - Return 1 if lhs is 1.
    // - Else return 0.
    Value ifLhsIsOne = lb.create<arith::SelectOp>(lhsIsOne, one, zero);
    Value ifLhsIsNegOne = lb.create<arith::SelectOp>(
        lhsIsNegOne, lb.create<arith::SelectOp>(rhsIsEven, one, negOne),
        ifLhsIsOne);
    return lb.create<arith::SelectOp>(rhsIsNegative, ifLhsIsNegOne, accum);
  } else if (IsFieldType{}(elementType)) {
    return b->create<zkir::field::PowUIOp>(loc, adaptor.getLhs(),
                                           adaptor.getRhs());
  }
  return nullptr;
}

}  // namespace impl

struct MhloOpToStdScalarOp {
  // Converts mhlo 'op' to linalg and arith ops.
  template <typename MhloOpTy>
  static Value mapOp(MhloOpTy op, ArrayRef<Type> resultTypes, ValueRange args,
                     ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    auto argTypes = llvm::to_vector(op->getOperandTypes());
    return mapOpWithArgTypes(op, resultTypes, argTypes, args, attributes, b);
  }

  // Converts mhlo 'op' to linalg and arith ops. The types of 'args' may already
  // be converted, 'argTypes' are their original types.
  template <typename MhloOpTy>
  static Value mapOpWithArgTypes(MhloOpTy op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 ArrayRef<NamedAttribute> attributes,
                                 OpBuilder* b) {
    static_assert(!std::is_same<MhloOpTy, mhlo::ConvertOp>::value);
    typename MhloOpTy::Adaptor adaptor(args, op->getAttrDictionary(),
                                       op->getPropertiesStorage(),
                                       op->getRegions());
    return mapOpOfType<MhloOpTy>(op.getLoc(), resultTypes, argTypes, adaptor,
                                 attributes, b);
  }
  // Overload for mhlo::ConvertOp.
  static Value mapOpWithArgTypes(mhlo::ConvertOp op, ArrayRef<Type> resultTypes,
                                 ArrayRef<Type> argTypes, ValueRange args,
                                 ArrayRef<NamedAttribute> attributes,
                                 OpBuilder* b) {
    return impl::mapConvertOpToStdScalarOp(
        op.getLoc(), op.getType(), resultTypes, argTypes, args, attributes, b);
  }

  // Converts mhlo 'op' to linalg and arith ops.
  template <typename MhloOpTy>
  static Value mapOpOfType(Location loc, ArrayRef<Type> resultTypes,
                           ArrayRef<Type> argTypes,
                           typename MhloOpTy::Adaptor adaptor,
                           ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    return impl::mapMhloOpToStdScalarOp<MhloOpTy>(loc, resultTypes, argTypes,
                                                  adaptor, attributes, b);
  }

  static Value mapConvertOpToStdScalarOp(
      Location loc, ArrayRef<Type> targetTypes, ArrayRef<Type> resultTypes,
      ArrayRef<Type> argTypes, ValueRange args,
      ArrayRef<NamedAttribute> attributes, OpBuilder* b) {
    return impl::mapConvertOpToStdScalarOp(loc, targetTypes, resultTypes,
                                           argTypes, args, attributes, b);
  }
};

}  // namespace mlir::mhlo

#endif  // ZKX_MLIR_HLO_MHLO_TRANSFORMS_MAP_MHLO_TO_SCALAR_OP_H_
