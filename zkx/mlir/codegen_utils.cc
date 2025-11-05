#include "zkx/mlir/codegen_utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Location.h"

namespace zkx::mlir_utils {

using namespace mlir;

Value ConvertInteger(ImplicitLocOpBuilder& b, ArrayRef<Type> result_types,
                     Type source_type, Type target_type, ValueRange args,
                     bool is_signed, ArrayRef<NamedAttribute> attributes) {
  if (target_type.isInteger(/*width=*/1)) {
    // When casting to bool, we need to compare whether the value is equal to
    // zero.
    Value zero =
        b.create<arith::ConstantOp>(b.getZeroAttr(args.front().getType()));
    return b.create<arith::CmpIOp>(arith::CmpIPredicate::ne, args.front(),
                                   zero);
  }
  auto src = cast<IntegerType>(source_type);
  auto res = cast<IntegerType>(target_type);
  if (src.getWidth() > res.getWidth()) {
    return b.create<arith::TruncIOp>(result_types, args, attributes);
  }
  if (src.getWidth() < res.getWidth()) {
    // Special case boolean values, so they get casted to `1` instead of `-1`.
    if (is_signed) {
      return b.create<arith::ExtSIOp>(result_types, args, attributes);
    } else {
      return b.create<arith::ExtUIOp>(result_types, args, attributes);
    }
  }
  // No conversion is needed for the same width integers
  return args.front();
}

Value PowerInteger(ImplicitLocOpBuilder& b, Value lhs, Value rhs,
                   bool is_signed) {
  // Exponentiation by squaring:
  // https://en.wikipedia.org/wiki/Exponentiation_by_squaring;
  Value neg_one =
      b.create<arith::ConstantOp>(b.getIntegerAttr(lhs.getType(), -1));
  Value zero = b.create<arith::ConstantOp>(b.getIntegerAttr(lhs.getType(), 0));
  Value one = b.create<arith::ConstantOp>(b.getIntegerAttr(lhs.getType(), 1));
  Value two = b.create<arith::ConstantOp>(b.getIntegerAttr(lhs.getType(), 2));
  Value step = b.create<arith::ConstantIndexOp>(1);
  Value lower_bound = b.create<arith::ConstantIndexOp>(0);
  // The number of iterations is chosen to be sufficient for the bit-width of
  // the exponent. For a 32-bit exponent, 32 iterations are needed.
  Value upper_bound = b.create<arith::ConstantIndexOp>(
      cast<IntegerType>(rhs.getType()).getWidth());
  Value original_base = lhs;
  Value original_exponent = rhs;

  Value accum =
      b.create<scf::ForOp>(
           lower_bound, upper_bound, step,
           SmallVector<Value>({one, original_base, original_exponent}),
           [&](OpBuilder& b, Location loc, Value /*v*/, ValueRange iters) {
             ImplicitLocOpBuilder nested_b(loc, b);
             Value accum = iters[0];
             Value base = iters[1];
             Value exponent = iters[2];

             Value condition = nested_b.create<arith::CmpIOp>(
                 arith::CmpIPredicate::eq,
                 nested_b.create<arith::AndIOp>(exponent, one), one);
             Value multiplied = nested_b.create<arith::MulIOp>(accum, base);
             accum =
                 nested_b.create<arith::SelectOp>(condition, multiplied, accum);
             base = nested_b.create<arith::MulIOp>(base, base);
             exponent = nested_b.create<arith::ShRUIOp>(exponent, one);
             nested_b.create<scf::YieldOp>(
                 SmallVector<Value>({accum, base, exponent}));
           })
          .getResult(0);

  if (!is_signed) {
    return accum;
  }

  Value rhs_is_even = b.create<arith::CmpIOp>(
      arith::CmpIPredicate::eq, b.create<arith::RemSIOp>(rhs, two), zero);
  Value rhs_is_negative =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, rhs, zero);
  Value lhs_is_one =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, one);
  Value lhs_is_neg_one =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lhs, neg_one);

  // The accum is correct when the rhs is non-negative. When rhs is
  // negative, we return 0 for integer, with the exception of lhs values of 1
  // and -1 which have integer results for negative exponents. Specifically, the
  // calculation is the following:
  //
  // - Return accum if the rhs is not negative.
  // - Return 1 or -1 depending on the parity of rhs when the lhs is -1.
  // - Return 1 if lhs is 1.
  // - Else return 0.
  Value if_lhs_is_one = b.create<arith::SelectOp>(lhs_is_one, one, zero);
  Value if_lhs_is_neg_one = b.create<arith::SelectOp>(
      lhs_is_neg_one, b.create<arith::SelectOp>(rhs_is_even, one, neg_one),
      if_lhs_is_one);
  return b.create<arith::SelectOp>(rhs_is_negative, if_lhs_is_neg_one, accum);
}

Value SignInteger(ImplicitLocOpBuilder& b, Value value) {
  // sign(x) = x == 0 ? 0 : ((x >>s (width - 1)) | 1)
  IntegerType integer_type = cast<IntegerType>(value.getType());
  Value zero = b.create<arith::ConstantOp>(b.getZeroAttr(integer_type));
  Value bit_width_minus_one = b.create<arith::ConstantOp>(
      b.getIntegerAttr(integer_type, integer_type.getWidth() - 1));
  Value one = b.create<arith::ConstantOp>(b.getOneAttr(integer_type));
  Value cmp = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, value, zero);
  Value shr = b.create<arith::ShRSIOp>(value, bit_width_minus_one);
  Value or_op = b.create<arith::OrIOp>(shr, one);
  return b.create<arith::SelectOp>(cmp, zero, or_op);
}

}  // namespace zkx::mlir_utils
