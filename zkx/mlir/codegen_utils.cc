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

}  // namespace zkx::mlir_utils
