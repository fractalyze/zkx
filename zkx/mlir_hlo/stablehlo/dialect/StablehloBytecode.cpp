/* Copyright 2022 The StableHLO Authors.
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

#include "zkx/mlir_hlo/stablehlo/dialect/StablehloBytecode.h"

#include <cstdint>
#include <memory>

#include "llvm/ADT/SmallVector.h" // IWYU pragma: keep
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "zkx/mlir_hlo/stablehlo/dialect/Base.h"
#include "zkx/mlir_hlo/stablehlo/dialect/StablehloOps.h"

//===----------------------------------------------------------------------===//
// Debug Trace Helpers
//===----------------------------------------------------------------------===//

// Enable logging with flag:
//   stablehlo-opt -debug-only=stablehlo-bytecode [...]
//
// Extract after function name, remove namespace.
//   Called: write(mlir::stablehlo::TokenType, mlir::DialectBytecodeWriter ...
//   ***Not Implemented: write(...
#define _EXTRACT_AFTER(a, b)                                                   \
  llvm::StringRef(a).substr(llvm::StringRef(a).find(b))

#define _LOG_CALL_TO(func)                                                     \
  DEBUG_WITH_TYPE("stablehlo-bytecode",                                        \
                  llvm::errs()                                                 \
                      << "Called: "                                            \
                      << _EXTRACT_AFTER(LLVM_PRETTY_FUNCTION, func) << '\n')

#define LOG_WRITE_CALL _LOG_CALL_TO("write")
#define LOG_READ_CALL _LOG_CALL_TO(__func__)
#define LOG_NOT_IMPLEMENTED                                                    \
  DEBUG_WITH_TYPE("stablehlo-bytecode", llvm::errs()                           \
                                            << "***Not Implemented: "          \
                                            << LLVM_PRETTY_FUNCTION << '\n')

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

namespace {
namespace stablehlo_encoding {

// This enum contains marker codes used to indicate which attribute is
// currently being decoded, and how it should be decoded. The order of these
// codes must not be changed, as any changes will break compatibility
// with older bytecode.
//
// To add an attribute, search for "TO ADD ATTRIBUTE" in this file and ensure
// each location is updated.
enum AttributeCode {
  // TO ADD ATTRIBUTE: Add an enum value with doc string for new attr.

  //   ComparisonDirectionAttr
  //     value: varint (encoded enum)
  //   }
  kComparisonDirectionAttr = 0,

  //   TypeExtensionsAttr {
  //     bounds : svarint[]
  //   }
  kTypeExtensionsAttr = 1,

  //   ScatterDimensionNumbersAttr {
  //     updateWindowDims: svarint[]
  //     insertedWindowDims: svarint[]
  //     scatterDimsToOperandDims: svarint[]
  //     indexVectorDim: svarint
  //   }
  kScatterDimensionNumbersAttr = 2,
};

// This enum contains marker codes used to indicate which type is
// currently being decoded, and how it should be decoded. The order of these
// codes must not be changed, as any changes will break compatibility
// with older bytecode.
///
// To add a type, search for "TO ADD TYPE" in this file and ensure each
// location is updated.
enum TypeCode {
  // TO ADD TYPE: Add an enum value with doc string for new type.

  //   TokenType {
  //   }
  kTokenType = 0,
};

} // namespace stablehlo_encoding
} // namespace

//===----------------------------------------------------------------------===//
// StablehloBytecodeInterface
//===----------------------------------------------------------------------===//

namespace mlir::stablehlo {
namespace {

// This class implements the bytecode interface for the StableHLO dialect.
class StablehloBytecodeInterface : public BytecodeDialectInterface {
public:
  StablehloBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes
  //===----------------------------------------------------------------------===//

  // These methods are invoked by superclass when an attr from StableHLO dialect
  // is encountered.
  Attribute readAttribute(DialectBytecodeReader &reader) const override;
  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override;

  // TO ADD ATTRIBUTE: Include a read method for each attribute in StableHLO
  // Ex: SomeAttr readSomeAttr(DialectBytecodeReader &reader) const;
  ComparisonDirectionAttr
  readComparisonDirectionAttr(DialectBytecodeReader &reader) const;
  TypeExtensionsAttr
  readTypeExtensionsAttr(DialectBytecodeReader &reader) const;
  ScatterDimensionNumbersAttr
  readScatterDimensionNumbersAttr(DialectBytecodeReader &reader) const;

  // TO ADD ATTRIBUTE: Include a write method for each attribute in StableHLO
  // Ex: void write(SomeAttr attr, DialectBytecodeWriter &writer) const;
  void write(ComparisonDirectionAttr attr, DialectBytecodeWriter &writer) const;
  void write(TypeExtensionsAttr attr, DialectBytecodeWriter &writer) const;
  void write(ScatterDimensionNumbersAttr attr,
             DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types
  //===----------------------------------------------------------------------===//

  // These methods are invoked by superclass when a type from StableHLO dialect
  // is encountered.
  Type readType(DialectBytecodeReader &reader) const override;
  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override;

  // TO ADD TYPE: Include a read method for each type in StableHLO
  // Ex: SomeType readSomeType(DialectBytecodeReader &reader) const;
  TokenType readTokenType(DialectBytecodeReader &reader) const;

  // TO ADD TYPE: Include a write method for each type in StableHLO
  // Ex: void write(SomeType attr, DialectBytecodeWriter &writer) const;
  void write(TokenType type, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Version
  //===----------------------------------------------------------------------===//

  std::unique_ptr<DialectVersion>
  readVersion(DialectBytecodeReader &reader) const final;

  void writeVersion(DialectBytecodeWriter &writer) const final;
};

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// TO ADD ATTRIBUTE: Update the switch to include a branch for the attr.
Attribute
StablehloBytecodeInterface::readAttribute(DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Attribute();
  switch (code) {
  case stablehlo_encoding::kComparisonDirectionAttr:
    return readComparisonDirectionAttr(reader);
  case stablehlo_encoding::kTypeExtensionsAttr:
    return readTypeExtensionsAttr(reader);
  case stablehlo_encoding::kScatterDimensionNumbersAttr:
    return readScatterDimensionNumbersAttr(reader);
  default:
    reader.emitError() << "unknown stablehlo attribute code: " << code;
    return Attribute();
  }
}

// TO ADD ATTRIBUTE: Update the case selection to include the new attr.
// If this method returns failure, the string serialization is used in the
// bytecode.
LogicalResult StablehloBytecodeInterface::writeAttribute(
    Attribute attr, DialectBytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<ComparisonDirectionAttr, TypeExtensionsAttr,
            ScatterDimensionNumbersAttr>([&](auto attr) {
        LOG_WRITE_CALL;
        write(attr, writer);
        return success();
      })
      .Default([&](Attribute) {
        LOG_NOT_IMPLEMENTED;
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// ComparisonDirectionAttr
//===----------------------------------------------------------------------===//

ComparisonDirectionAttr StablehloBytecodeInterface::readComparisonDirectionAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonDirectionAttr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonDirection(val); });
}

void StablehloBytecodeInterface::write(ComparisonDirectionAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kComparisonDirectionAttr);
  hlo::bytecode::writeEnumAttribute<ComparisonDirection>(attr, writer);
}

//===----------------------------------------------------------------------===//
// TypeExtensionsAttr
//===----------------------------------------------------------------------===//

TypeExtensionsAttr StablehloBytecodeInterface::readTypeExtensionsAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> bounds;
  if (failed(reader.readSignedVarInts(bounds)))
    return TypeExtensionsAttr();
  return TypeExtensionsAttr::get(getContext(), bounds);
}

void StablehloBytecodeInterface::write(TypeExtensionsAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kTypeExtensionsAttr);
  writer.writeSignedVarInts(attr.getBounds());
}

//===----------------------------------------------------------------------===//
// ScatterDimensionNumbersAttr
//===----------------------------------------------------------------------===//

ScatterDimensionNumbersAttr
StablehloBytecodeInterface::readScatterDimensionNumbersAttr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> updateWindowDims, insertedWindowDims,
      inputBatchingDims, scatterIndicesBatchingDims, scatterDimsToOperandDims;
  int64_t indexVectorDim;

  if (failed(reader.readSignedVarInts(updateWindowDims)) ||
      failed(reader.readSignedVarInts(insertedWindowDims)) ||
      failed(reader.readSignedVarInts(inputBatchingDims)) ||
      failed(reader.readSignedVarInts(scatterIndicesBatchingDims)) ||
      failed(reader.readSignedVarInts(scatterDimsToOperandDims)) ||
      failed(reader.readSignedVarInt(indexVectorDim)))
    return ScatterDimensionNumbersAttr();

  return ScatterDimensionNumbersAttr::get(
      getContext(), updateWindowDims, insertedWindowDims, inputBatchingDims,
      scatterIndicesBatchingDims, scatterDimsToOperandDims, indexVectorDim);
}

void StablehloBytecodeInterface::write(ScatterDimensionNumbersAttr attr,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kScatterDimensionNumbersAttr);
  writer.writeSignedVarInts(attr.getUpdateWindowDims());
  writer.writeSignedVarInts(attr.getInsertedWindowDims());
  writer.writeSignedVarInts(attr.getInputBatchingDims());
  writer.writeSignedVarInts(attr.getScatterIndicesBatchingDims());
  writer.writeSignedVarInts(attr.getScatterDimsToOperandDims());
  writer.writeSignedVarInt(attr.getIndexVectorDim());
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// TO ADD TYPE: Update the case selection to include the new type.
Type StablehloBytecodeInterface::readType(DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Type();

  switch (code) {
  case stablehlo_encoding::kTokenType:
    return readTokenType(reader);

  default:
    reader.emitError() << "unknown builtin type code: " << code;
    return Type();
  }
}

LogicalResult
StablehloBytecodeInterface::writeType(Type type,
                                      DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<TokenType>([&](auto type) {
        LOG_WRITE_CALL;
        write(type, writer);
        return success();
      })
      .Default([&](Type) {
        LOG_NOT_IMPLEMENTED;
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// TokenType
//===----------------------------------------------------------------------===//

TokenType
StablehloBytecodeInterface::readTokenType(DialectBytecodeReader &) const {
  LOG_READ_CALL;
  return TokenType::get(getContext());
}

void StablehloBytecodeInterface::write(TokenType type,
                                       DialectBytecodeWriter &writer) const {
  writer.writeVarInt(stablehlo_encoding::kTokenType);
}

std::unique_ptr<DialectVersion>
StablehloBytecodeInterface::readVersion(DialectBytecodeReader &reader) const {
  uint64_t major, minor, patch;
  if (failed(reader.readVarInt(major)) || failed(reader.readVarInt(minor)) ||
      failed(reader.readVarInt(patch)))
    return nullptr;

  auto version = std::make_unique<StablehloDialectVersion>(major, minor, patch);
  if (version && StablehloDialectVersion::getCurrentVersion() < *version) {
    // Note: dialect bytecode reader does not expose emitWarning.
    // TODO(jpienaar): Update when it does.
    mlir::emitWarning(mlir::UnknownLoc::get(getContext()))
        << "reading newer dialect than supported";
    return nullptr;
  }

  return version;
}

void StablehloBytecodeInterface::writeVersion(
    DialectBytecodeWriter &writer) const {
  if (auto version = cast<StablehloDialect>(getDialect())->getVersion()) {
    writer.writeVarInt(static_cast<uint64_t>(version->getMajor()));
    writer.writeVarInt(static_cast<uint64_t>(version->getMinor()));
    writer.writeVarInt(static_cast<uint64_t>(version->getPatch()));
  }
}

} // namespace

void addBytecodeInterface(StablehloDialect *dialect) {
  dialect->addInterfaces<StablehloBytecodeInterface>();
}

} // namespace mlir::stablehlo
