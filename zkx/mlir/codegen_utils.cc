#include "zkx/mlir/codegen_utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

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
