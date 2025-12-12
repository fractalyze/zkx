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

#include "zkx/mlir_hlo/stablehlo/dialect/VhloBytecode.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "zkx/mlir_hlo/stablehlo/dialect/Base.h"
#include "zkx/mlir_hlo/stablehlo/dialect/VhloOps.h"
#include "zkx/mlir_hlo/stablehlo/dialect/VhloTypes.h"

//===----------------------------------------------------------------------===//
// Debug Trace Helpers
//===----------------------------------------------------------------------===//

// Enable logging with flag:
//   stablehlo-opt -debug-only=vhlo-bytecode [...]
//
// Extract after function name, remove namespace.
//   Called: write(mlir::vhlo::TokenV1Type, mlir::DialectBytecodeWriter ...
//   ***Not Implemented: write(...
#define _EXTRACT_AFTER(a, b)                                                   \
  llvm::StringRef(a).substr(llvm::StringRef(a).find(b))

#define DEBUG_TYPE "vhlo-bytecode"

#define _LOG_CALL_TO(func)                                                     \
  LLVM_DEBUG(llvm::errs() << "Called: "                                        \
                          << _EXTRACT_AFTER(LLVM_PRETTY_FUNCTION, func)        \
                          << '\n')

#define LOG_WRITE_CALL _LOG_CALL_TO("write")
#define LOG_READ_CALL _LOG_CALL_TO(__func__)
#define LOG_NOT_IMPLEMENTED(typeOrAttr)                                        \
  LLVM_DEBUG(llvm::errs() << "***Not Implemented: " << typeOrAttr << " "       \
                          << LLVM_PRETTY_FUNCTION << '\n')

//===----------------------------------------------------------------------===//
// Encoding
//===----------------------------------------------------------------------===//

namespace {
namespace vhlo_encoding {

// This enum contains marker codes used to indicate which attribute is
// currently being decoded, and how it should be decoded. The order of these
// codes must not be changed, as any changes will break compatibility
// with older bytecode.
//
// To add an attribute, search for "TO ADD ATTRIBUTE" in this file and ensure
// each location is updated.
enum AttributeCode {
  // TO ADD ATTRIBUTE: Add an enum value with doc string for new attr.

  //   ArrayV1Attr {
  //     elements: Attribute[]
  //   }
  kArrayV1Attr = 0,

  //   BooleanV1Attr {
  //     value: varint
  //   }
  kBooleanV1Attr = 1,

  //   ComparisonDirectionV1Attr
  //     value: varint (encoded enum)
  //   }
  kComparisonDirectionV1Attr = 2,

  //   DictionaryV1Attr {
  //     attrs: <Attribute, Attribute>[]
  //   }
  kDictionaryV1Attr = 3,

  //   IntegerV1Attr {
  //     type: Type
  //     value: APInt
  //   }
  kIntegerV1Attr = 4,

  //   StringV1Attr {
  //     value: string
  //   }
  kStringV1Attr = 5,

  //   TensorV1Attr {
  //     type: Type
  //     data: blob
  //   }
  kTensorV1Attr = 6,

  //   TypeV1Attr {
  //     value: Type
  //   }
  kTypeV1Attr = 7,

  //   TypeExtensionsV1Attr {
  //     bounds : svarint[]
  //   }
  kTypeExtensionsV1Attr = 8,
};

// This enum contains marker codes used to indicate which type is
// currently being decoded, and how it should be decoded. The order of these
// codes must not be changed, as any changes will break compatibility
// with older bytecode.
//
// To add a type, search for "TO ADD TYPE" in this file and ensure each
// location is updated.
enum TypeCode {
  // TO ADD TYPE: Add an enum value with doc string for new type.
  // Next available code: 41

  //   BooleanV1Type {
  //   }
  kBooleanV1Type = 0,

  //   FunctionV1Type {
  //     inputs: Type[]
  //     outputs: Type[]
  //   }
  kFunctionV1Type = 1,

  //   IndexV1Type {
  //   }
  kIndexV1Type = 2,

  //   IntegerSI2V1Type {
  //   }
  kIntegerSI2V1Type = 3,

  //   IntegerSI4V1Type {
  //   }
  kIntegerSI4V1Type = 4,

  //   IntegerSI8V1Type {
  //   }
  kIntegerSI8V1Type = 5,

  //   IntegerSI16V1Type {
  //   }
  kIntegerSI16V1Type = 6,

  //   IntegerSI32V1Type {
  //   }
  kIntegerSI32V1Type = 7,

  //   IntegerSI64V1Type {
  //   }
  kIntegerSI64V1Type = 8,

  //   IntegerUI2V1Type {
  //   }
  kIntegerUI2V1Type = 9,

  //   IntegerUI4V1Type {
  //   }
  kIntegerUI4V1Type = 10,

  //   IntegerUI8V1Type {
  //   }
  kIntegerUI8V1Type = 11,

  //   IntegerUI16V1Type {
  //   }
  kIntegerUI16V1Type = 12,

  //   IntegerUI32V1Type {
  //   }
  kIntegerUI32V1Type = 13,

  //   IntegerUI64V1Type {
  //   }
  kIntegerUI64V1Type = 14,

  //   RankedTensorV1Type {
  //     shape: svarint[],
  //     elementType: Type
  //   }
  kRankedTensorV1Type = 15,

  //   RankedTensorV1TypeWithEncoding {
  //     encoding: Attribute,
  //     shape: svarint[],
  //     elementType: Type
  //   }
  kRankedTensorV1TypeWithEncoding = 16,

  //   TokenV1Type {
  //   }
  kTokenV1Type = 17,

  //   TupleV1Type {
  //     elementTypes: Type[]
  //   }
  kTupleV1Type = 18,

  //   UnrankedTensorV1Type {
  //     elementType: Type
  //   }
  kUnrankedTensorV1Type = 19,

  //   WitnessV1Type {
  //   }
  kWitnessV1Type = 20,

  // NoneV1Type {
  // }
  kNoneV1Type = 21,
};

} // namespace vhlo_encoding
} // namespace

//===----------------------------------------------------------------------===//
// VhloBytecodeInterface
//===----------------------------------------------------------------------===//

namespace mlir {
namespace vhlo {
namespace {

// This class implements the bytecode interface for the VHLO dialect.
class VhloBytecodeInterface : public BytecodeDialectInterface {
public:
  VhloBytecodeInterface(Dialect *dialect) : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  // These methods are invoked by superclass when an attr from VHLO dialect
  // is encountered.
  Attribute readAttribute(DialectBytecodeReader &reader) const override;
  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override;

  // TO ADD ATTRIBUTE: Include a read method for each attribute in VHLO
  // Ex: SomeAttr readSomeAttr(DialectBytecodeReader &reader) const;
  ArrayV1Attr readArrayV1Attr(DialectBytecodeReader &reader) const;
  BooleanV1Attr readBooleanV1Attr(DialectBytecodeReader &reader) const;
  ComparisonDirectionV1Attr
  readComparisonDirectionV1Attr(DialectBytecodeReader &reader) const;
  DictionaryV1Attr readDictionaryV1Attr(DialectBytecodeReader &reader) const;
  IntegerV1Attr readIntegerV1Attr(DialectBytecodeReader &reader) const;
  StringV1Attr readStringV1Attr(DialectBytecodeReader &reader) const;
  TensorV1Attr readTensorV1Attr(DialectBytecodeReader &reader) const;
  TypeV1Attr readTypeV1Attr(DialectBytecodeReader &reader) const;
  TypeExtensionsV1Attr
  readTypeExtensionsV1Attr(DialectBytecodeReader &reader) const;

  // TO ADD ATTRIBUTE: Include a write method for each attribute in VHLO
  // Ex: void write(SomeAttr attr, DialectBytecodeWriter &writer) const;
  void write(ArrayV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(BooleanV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(ComparisonDirectionV1Attr attr,
             DialectBytecodeWriter &writer) const;
  void write(DictionaryV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(IntegerV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(StringV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(TensorV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(TypeV1Attr attr, DialectBytecodeWriter &writer) const;
  void write(TypeExtensionsV1Attr attr, DialectBytecodeWriter &writer) const;

  //===--------------------------------------------------------------------===//
  // Types

  // These methods are invoked by superclass when a type from VHLO dialect
  // is encountered.
  Type readType(DialectBytecodeReader &reader) const override;
  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override;

  // TO ADD TYPE: Include a read method for each type in VHLO
  // Ex: SomeType readSomeType(DialectBytecodeReader &reader) const;
  FunctionV1Type readFunctionV1Type(DialectBytecodeReader &reader) const;
  RankedTensorV1Type readRankedTensorV1Type(DialectBytecodeReader &reader,
                                            bool hasEncoding) const;
  TokenV1Type readTokenV1Type(DialectBytecodeReader &reader) const;
  TupleV1Type readTupleV1Type(DialectBytecodeReader &reader) const;
  UnrankedTensorV1Type
  readUnrankedTensorV1Type(DialectBytecodeReader &reader) const;

  // TO ADD TYPE: Include a write method for each type in VHLO
  // Ex: void write(SomeType attr, DialectBytecodeWriter &writer) const;
  void write(FunctionV1Type type, DialectBytecodeWriter &writer) const;
  void write(RankedTensorV1Type type, DialectBytecodeWriter &writer) const;
  void write(TokenV1Type type, DialectBytecodeWriter &writer) const;
  void write(TupleV1Type type, DialectBytecodeWriter &writer) const;
  void write(UnrankedTensorV1Type type, DialectBytecodeWriter &writer) const;
};

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// TO ADD ATTRIBUTE: Update the switch to include a branch for the attr.
Attribute
VhloBytecodeInterface::readAttribute(DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Attribute();
  switch (code) {
  case vhlo_encoding::kArrayV1Attr:
    return readArrayV1Attr(reader);
  case vhlo_encoding::kBooleanV1Attr:
    return readBooleanV1Attr(reader);
  case vhlo_encoding::kComparisonDirectionV1Attr:
    return readComparisonDirectionV1Attr(reader);
  case vhlo_encoding::kDictionaryV1Attr:
    return readDictionaryV1Attr(reader);
  case vhlo_encoding::kIntegerV1Attr:
    return readIntegerV1Attr(reader);
  case vhlo_encoding::kStringV1Attr:
    return readStringV1Attr(reader);
  case vhlo_encoding::kTensorV1Attr:
    return readTensorV1Attr(reader);
  case vhlo_encoding::kTypeV1Attr:
    return readTypeV1Attr(reader);
  case vhlo_encoding::kTypeExtensionsV1Attr:
    return readTypeExtensionsV1Attr(reader);
  default:
    reader.emitError() << "unknown vhlo attribute code: " << code;
    return Attribute();
  }
}

// TO ADD ATTRIBUTE: Update the case selection to include the new attr.
// If this method returns failure, the string serialization is used in the
// bytecode.
LogicalResult
VhloBytecodeInterface::writeAttribute(Attribute attr,
                                      DialectBytecodeWriter &writer) const {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<ArrayV1Attr, BooleanV1Attr, ComparisonDirectionV1Attr,
            DictionaryV1Attr, IntegerV1Attr, StringV1Attr, TensorV1Attr,
            TypeV1Attr, TypeExtensionsV1Attr>([&](auto attr) {
        LOG_WRITE_CALL;
        write(attr, writer);
        return success();
      })
      .Default([&](Attribute attr) {
        LOG_NOT_IMPLEMENTED(attr);
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// ArrayV1Attr
//===----------------------------------------------------------------------===//

ArrayV1Attr
VhloBytecodeInterface::readArrayV1Attr(DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  SmallVector<Attribute> elements;
  if (failed(reader.readAttributes(elements)))
    return ArrayV1Attr();
  return ArrayV1Attr::get(getContext(), elements);
}

void VhloBytecodeInterface::write(ArrayV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kArrayV1Attr);
  writer.writeAttributes(attr.getValue());
}

//===----------------------------------------------------------------------===//
// BooleanV1Attr
//===----------------------------------------------------------------------===//

BooleanV1Attr
VhloBytecodeInterface::readBooleanV1Attr(DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  uint64_t int_value;
  if (failed(reader.readVarInt(int_value)))
    return BooleanV1Attr();
  if (int_value != 0 && int_value != 1) {
    reader.emitError() << "unsupported value: " << int_value;
    return BooleanV1Attr();
  }
  return BooleanV1Attr::get(getContext(), int_value == 1);
}

void VhloBytecodeInterface::write(BooleanV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kBooleanV1Attr);
  writer.writeVarInt(attr.getValue() ? 1 : 0);
}

//===----------------------------------------------------------------------===//
// ComparisonDirectionV1Attr
//===----------------------------------------------------------------------===//

ComparisonDirectionV1Attr VhloBytecodeInterface::readComparisonDirectionV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  return hlo::bytecode::readEnumAttribute<ComparisonDirectionV1Attr>(
      reader, getContext(),
      [](uint32_t val) { return symbolizeComparisonDirectionV1(val); });
}

void VhloBytecodeInterface::write(ComparisonDirectionV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kComparisonDirectionV1Attr);
  hlo::bytecode::writeEnumAttribute<ComparisonDirectionV1>(attr, writer);
}

//===----------------------------------------------------------------------===//
// DictionaryV1Attr
//===----------------------------------------------------------------------===//

DictionaryV1Attr VhloBytecodeInterface::readDictionaryV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  auto readNamedAttr = [&]() -> FailureOr<std::pair<Attribute, Attribute>> {
    Attribute name;
    Attribute value;
    if (failed(reader.readAttribute(name)) ||
        failed(reader.readAttribute(value)))
      return failure();
    return {{name, value}};
  };
  SmallVector<std::pair<Attribute, Attribute>> attrs;
  if (failed(reader.readList(attrs, readNamedAttr)))
    return DictionaryV1Attr();

  return DictionaryV1Attr::get(getContext(), attrs);
}

void VhloBytecodeInterface::write(DictionaryV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kDictionaryV1Attr);
  writer.writeList(attr.getValue(), [&](auto attrPair) {
    writer.writeAttribute(attrPair.first);
    writer.writeAttribute(attrPair.second);
  });
}

//===----------------------------------------------------------------------===//
// IntegerV1Attr
//===----------------------------------------------------------------------===//

namespace {

std::optional<unsigned> getBitWidthForIntegerType(Type type) {
  if (isa<IntegerSI2V1Type>(type) || isa<IntegerUI2V1Type>(type))
    return 2;
  if (isa<IntegerSI4V1Type>(type) || isa<IntegerUI4V1Type>(type))
    return 4;
  if (isa<IntegerSI8V1Type>(type) || isa<IntegerUI8V1Type>(type))
    return 8;
  if (isa<IntegerSI16V1Type>(type) || isa<IntegerUI16V1Type>(type))
    return 16;
  if (isa<IntegerSI32V1Type>(type) || isa<IntegerUI32V1Type>(type))
    return 32;
  if (isa<IntegerSI64V1Type>(type) || isa<IntegerUI64V1Type>(type))
    return 64;
  return std::nullopt;
}

} // namespace

IntegerV1Attr
VhloBytecodeInterface::readIntegerV1Attr(DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type type;
  if (failed(reader.readType(type)))
    return IntegerV1Attr();

  // Extract the value storage width from the type.
  std::optional<unsigned> bitWidth;
  if (isa<IndexV1Type>(type)) {
    bitWidth = IndexType::kInternalStorageBitWidth;
  } else {
    bitWidth = getBitWidthForIntegerType(type);
  }

  if (!bitWidth) {
    reader.emitError() << "unsupported integer type for IntegerV1Attr: "
                       << type;
    return IntegerV1Attr();
  }

  FailureOr<APInt> value = reader.readAPIntWithKnownWidth(*bitWidth);
  if (failed(value))
    return IntegerV1Attr();
  return IntegerV1Attr::get(getContext(), type, *value);
}

void VhloBytecodeInterface::write(IntegerV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kIntegerV1Attr);
  writer.writeType(attr.getType());
  writer.writeAPIntWithKnownWidth(attr.getValue());
}

//===----------------------------------------------------------------------===//
// StringV1Attr
//===----------------------------------------------------------------------===//

StringV1Attr
VhloBytecodeInterface::readStringV1Attr(DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  StringRef string;
  if (failed(reader.readString(string)))
    return StringV1Attr();
  return StringV1Attr::get(getContext(), string);
}

void VhloBytecodeInterface::write(StringV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kStringV1Attr);
  writer.writeOwnedString(attr.getValue());
}

//===----------------------------------------------------------------------===//
// TensorV1Attr
//===----------------------------------------------------------------------===//

TensorV1Attr
VhloBytecodeInterface::readTensorV1Attr(DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type type;
  ArrayRef<char> blob;
  if (failed(reader.readType(type)) || failed(reader.readBlob(blob)))
    return TensorV1Attr();
  return TensorV1Attr::get(getContext(), type, blob);
}

void VhloBytecodeInterface::write(TensorV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTensorV1Attr);
  writer.writeType(attr.getType());
  writer.writeOwnedBlob(attr.getData());
}

//===----------------------------------------------------------------------===//
// TypeV1Attr
//===----------------------------------------------------------------------===//

TypeV1Attr
VhloBytecodeInterface::readTypeV1Attr(DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type type;
  if (failed(reader.readType(type)))
    return TypeV1Attr();

  return TypeV1Attr::get(getContext(), type);
}

void VhloBytecodeInterface::write(TypeV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTypeV1Attr);
  writer.writeType(attr.getValue());
}

//===----------------------------------------------------------------------===//
// TypeExtensionsV1Attr
//===----------------------------------------------------------------------===//

TypeExtensionsV1Attr VhloBytecodeInterface::readTypeExtensionsV1Attr(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  llvm::SmallVector<int64_t> bounds;
  if (failed(reader.readSignedVarInts(bounds)))
    return TypeExtensionsV1Attr();
  return TypeExtensionsV1Attr::get(getContext(), bounds);
}

void VhloBytecodeInterface::write(TypeExtensionsV1Attr attr,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTypeExtensionsV1Attr);
  writer.writeSignedVarInts(attr.getBounds());
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// TO ADD TYPE: Update the case selection to include the new type.
Type VhloBytecodeInterface::readType(DialectBytecodeReader &reader) const {
  uint64_t code;
  if (failed(reader.readVarInt(code)))
    return Type();

  switch (code) {
  case vhlo_encoding::kBooleanV1Type:
    return BooleanV1Type::get(getContext());
  case vhlo_encoding::kFunctionV1Type:
    return readFunctionV1Type(reader);
  case vhlo_encoding::kIndexV1Type:
    return IndexV1Type::get(getContext());
  case vhlo_encoding::kIntegerSI2V1Type:
    return IntegerSI2V1Type::get(getContext());
  case vhlo_encoding::kIntegerSI4V1Type:
    return IntegerSI4V1Type::get(getContext());
  case vhlo_encoding::kIntegerSI8V1Type:
    return IntegerSI8V1Type::get(getContext());
  case vhlo_encoding::kIntegerSI16V1Type:
    return IntegerSI16V1Type::get(getContext());
  case vhlo_encoding::kIntegerSI32V1Type:
    return IntegerSI32V1Type::get(getContext());
  case vhlo_encoding::kIntegerSI64V1Type:
    return IntegerSI64V1Type::get(getContext());
  case vhlo_encoding::kIntegerUI2V1Type:
    return IntegerUI2V1Type::get(getContext());
  case vhlo_encoding::kIntegerUI4V1Type:
    return IntegerUI4V1Type::get(getContext());
  case vhlo_encoding::kIntegerUI8V1Type:
    return IntegerUI8V1Type::get(getContext());
  case vhlo_encoding::kIntegerUI16V1Type:
    return IntegerUI16V1Type::get(getContext());
  case vhlo_encoding::kIntegerUI32V1Type:
    return IntegerUI32V1Type::get(getContext());
  case vhlo_encoding::kIntegerUI64V1Type:
    return IntegerUI64V1Type::get(getContext());
  case vhlo_encoding::kNoneV1Type:
    return NoneV1Type::get(getContext());
  case vhlo_encoding::kRankedTensorV1Type:
    return readRankedTensorV1Type(reader, /*hasEncoding=*/false);
  case vhlo_encoding::kRankedTensorV1TypeWithEncoding:
    return readRankedTensorV1Type(reader, /*hasEncoding=*/true);
  case vhlo_encoding::kTokenV1Type:
    return readTokenV1Type(reader);
  case vhlo_encoding::kTupleV1Type:
    return readTupleV1Type(reader);
  case vhlo_encoding::kUnrankedTensorV1Type:
    return readUnrankedTensorV1Type(reader);
  case vhlo_encoding::kWitnessV1Type:
    return WitnessV1Type::get(getContext());
  default:
    reader.emitError() << "unknown vhlo type code: " << code;
    return Type();
  }
}

// TO ADD TYPE: Update the case selection to include the new type.
LogicalResult
VhloBytecodeInterface::writeType(Type type,
                                 DialectBytecodeWriter &writer) const {
  return TypeSwitch<Type, LogicalResult>(type)
      .Case<FunctionV1Type, RankedTensorV1Type, TokenV1Type, TupleV1Type,
            UnrankedTensorV1Type>([&](auto type) {
        LOG_WRITE_CALL;
        return write(type, writer), success();
      })
      .Case([&](BooleanV1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kBooleanV1Type), success();
      })
      .Case([&](IndexV1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIndexV1Type), success();
      })
      .Case([&](IntegerSI2V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerSI2V1Type), success();
      })
      .Case([&](IntegerSI4V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerSI4V1Type), success();
      })
      .Case([&](IntegerSI8V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerSI8V1Type), success();
      })
      .Case([&](IntegerSI16V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerSI16V1Type), success();
      })
      .Case([&](IntegerSI32V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerSI32V1Type), success();
      })
      .Case([&](IntegerSI64V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerSI64V1Type), success();
      })
      .Case([&](IntegerUI2V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI2V1Type), success();
      })
      .Case([&](IntegerUI4V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI4V1Type), success();
      })
      .Case([&](IntegerUI8V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI8V1Type), success();
      })
      .Case([&](IntegerUI16V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI16V1Type), success();
      })
      .Case([&](IntegerUI32V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI32V1Type), success();
      })
      .Case([&](IntegerUI64V1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kIntegerUI64V1Type), success();
      })
      .Case([&](NoneV1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kNoneV1Type), success();
      })
      .Case([&](WitnessV1Type) {
        LOG_WRITE_CALL;
        return writer.writeVarInt(vhlo_encoding::kWitnessV1Type), success();
      })
      .Default([&](Type type) {
        LOG_NOT_IMPLEMENTED(type);
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// FunctionV1Type
//===----------------------------------------------------------------------===//

FunctionV1Type
VhloBytecodeInterface::readFunctionV1Type(DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  SmallVector<Type> inputs;
  SmallVector<Type> outputs;
  if (failed(reader.readTypes(inputs)) || failed(reader.readTypes(outputs)))
    return FunctionV1Type();

  return FunctionV1Type::get(getContext(), inputs, outputs);
}

void VhloBytecodeInterface::write(FunctionV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kFunctionV1Type);
  writer.writeTypes(type.getInputs());
  writer.writeTypes(type.getOutputs());
}

//===----------------------------------------------------------------------===//
// RankedTensorV1Type
//===----------------------------------------------------------------------===//

RankedTensorV1Type
VhloBytecodeInterface::readRankedTensorV1Type(DialectBytecodeReader &reader,
                                              bool hasEncoding) const {
  LOG_READ_CALL;
  Attribute encoding;
  if (hasEncoding && failed(reader.readAttribute(encoding)))
    return RankedTensorV1Type();

  SmallVector<int64_t> shape;
  Type elementType;
  if (failed(reader.readSignedVarInts(shape)) ||
      failed(reader.readType(elementType)))
    return RankedTensorV1Type();

  return RankedTensorV1Type::get(getContext(), shape, elementType, encoding);
}

void VhloBytecodeInterface::write(RankedTensorV1Type type,
                                  DialectBytecodeWriter &writer) const {
  if (Attribute encoding = type.getEncoding()) {
    writer.writeVarInt(vhlo_encoding::kRankedTensorV1TypeWithEncoding);
    writer.writeAttribute(encoding);
  } else {
    writer.writeVarInt(vhlo_encoding::kRankedTensorV1Type);
  }
  writer.writeSignedVarInts(type.getShape());
  writer.writeType(type.getElementType());
}

//===----------------------------------------------------------------------===//
// TokenV1Type
//===----------------------------------------------------------------------===//

TokenV1Type
VhloBytecodeInterface::readTokenV1Type(DialectBytecodeReader &) const {
  LOG_READ_CALL;
  return TokenV1Type::get(getContext());
}

void VhloBytecodeInterface::write(TokenV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTokenV1Type);
}

//===----------------------------------------------------------------------===//
// TupleV1Type
//===----------------------------------------------------------------------===//

TupleV1Type
VhloBytecodeInterface::readTupleV1Type(DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  SmallVector<Type> elements;
  if (failed(reader.readTypes(elements)))
    return TupleV1Type();

  return TupleV1Type::get(getContext(), elements);
}

void VhloBytecodeInterface::write(TupleV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kTupleV1Type);
  writer.writeTypes(type.getTypes());
}

//===----------------------------------------------------------------------===//
// UnrankedTensorV1Type
//===----------------------------------------------------------------------===//

UnrankedTensorV1Type VhloBytecodeInterface::readUnrankedTensorV1Type(
    DialectBytecodeReader &reader) const {
  LOG_READ_CALL;
  Type elementType;
  if (failed(reader.readType(elementType)))
    return UnrankedTensorV1Type();

  return UnrankedTensorV1Type::get(getContext(), elementType);
}

void VhloBytecodeInterface::write(UnrankedTensorV1Type type,
                                  DialectBytecodeWriter &writer) const {
  writer.writeVarInt(vhlo_encoding::kUnrankedTensorV1Type);
  writer.writeType(type.getElementType());
}

} // namespace

void addBytecodeInterface(VhloDialect *dialect) {
  dialect->addInterfaces<VhloBytecodeInterface>();
}

} // namespace vhlo
} // namespace mlir
