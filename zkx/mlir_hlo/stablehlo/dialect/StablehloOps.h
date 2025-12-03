/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.
   Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLOOPS_H_
#define ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLOOPS_H_

#include <optional>

#include "llvm/ADT/StringRef.h" // IWYU pragma: keep
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TensorEncoding.h" // IWYU pragma: keep
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h" // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h" // IWYU pragma: keep
#include "mlir/Support/LogicalResult.h"           // IWYU pragma: keep

#include "zkx/mlir_hlo/stablehlo/dialect/Base.h" // IWYU pragma: keep
#include "zkx/mlir_hlo/stablehlo/dialect/Version.h"

#define GET_TYPEDEF_CLASSES
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloTypeDefs.h.inc"

// Include order matters.
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloAttrs.h.inc"

namespace mlir::stablehlo {

struct StablehloDialectVersion : public mlir::DialectVersion {
  StablehloDialectVersion(int64_t major, int64_t minor, int64_t patch)
      : dialectVersion(major, minor, patch) {}

  int64_t getMajor() const { return dialectVersion.getMajor(); }
  int64_t getMinor() const { return dialectVersion.getMinor(); }
  int64_t getPatch() const { return dialectVersion.getPatch(); }

  static StablehloDialectVersion getCurrentVersion() {
    // The same version as VHLO as this is serialization related only.
    auto vhloVer = vhlo::Version::getCurrentVersion();
    return {vhloVer.getMajor(), vhloVer.getMinor(), vhloVer.getPatch()};
  }

  bool operator<(const StablehloDialectVersion &other) const {
    return this->dialectVersion < other.dialectVersion;
  }

private:
  // The dialect version read from bytecode.
  vhlo::Version dialectVersion;
};

class StablehloDialect : public Dialect {
public:
  explicit StablehloDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "stablehlo"; }

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  // Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;

  // Parses an attribute registered to this dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  // Prints an attribute registered to this dialect.
  void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;

  // Get the set dialect version.
  std::optional<StablehloDialectVersion> getVersion() const;

  // Set dialect version.
  // Note: there is currently no validation.
  void setVersion(std::optional<StablehloDialectVersion> version);

private:
  std::optional<StablehloDialectVersion> version;
};

} // namespace mlir::stablehlo

#define GET_OP_CLASSES
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloOps.h.inc"

namespace mlir::stablehlo {

// Returns the broadcast_dimensions for a BroadcastInDimOp from the
// result_type and broadcast_sizes from a BroadcastOp.
DenseI64ArrayAttr
getBroadcastDimensionsFromBroadcastSizes(RankedTensorType resultType,
                                         DenseI64ArrayAttr broadcastSizes);

SortOp createSortOp(PatternRewriter *rewriter, const Location &loc,
                    const llvm::ArrayRef<Value> &operands,
                    const llvm::ArrayRef<Type> &elementTypes, int64_t dimension,
                    bool isStable, ComparisonDirection direction);

} // namespace mlir::stablehlo

#endif // ZKX_MLIR_HLO_STABLEHLO_DIALECT_STABLEHLOOPS_H_
