#ifndef ZKX_MLIR_CODEGEN_UTILS_H_
#define ZKX_MLIR_CODEGEN_UTILS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace zkx::mlir_utils {

mlir::Value ConvertInteger(
    mlir::ImplicitLocOpBuilder& b, mlir::ArrayRef<mlir::Type> result_types,
    mlir::Type source_type, mlir::Type target_type, mlir::ValueRange args,
    bool is_signed, mlir::ArrayRef<mlir::NamedAttribute> attributes = {});

mlir::Value PowerInteger(mlir::ImplicitLocOpBuilder& b, mlir::Value lhs,
                         mlir::Value rhs, bool is_signed);

mlir::Value SignInteger(mlir::ImplicitLocOpBuilder& b, mlir::Value value);

}  // namespace zkx::mlir_utils

#endif  // ZKX_MLIR_CODEGEN_UTILS_H_
