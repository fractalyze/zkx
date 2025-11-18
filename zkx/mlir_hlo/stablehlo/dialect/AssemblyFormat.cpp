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

#include "zkx/mlir_hlo/stablehlo/dialect/AssemblyFormat.h"

#include <assert.h>

#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"

#include "zkx/mlir_hlo/stablehlo/dialect/Base.h"

#define DEBUG_TYPE "hlo-assembly"

namespace mlir::hlo {

//===----------------------------------------------------------------------===//
// Generic Type Printer and Parser
//===----------------------------------------------------------------------===//
namespace {

// Utility function, used by printSelectOpType and
// printSameOperandsAndResultType. Given a FunctionType, assign the types
// to operands and results, erroring if any mismatch in number of operands
// or results occurs.
ParseResult assignFromFunctionType(OpAsmParser& parser, llvm::SMLoc loc,
                                   ArrayRef<Type*> operands, Type& result,
                                   FunctionType& fnType) {
  assert(fnType);
  if (fnType.getInputs().size() != operands.size())
    return parser.emitError(loc)
           << operands.size() << " operands present, but expected "
           << fnType.getInputs().size();

  // Set operand types to function input types
  for (auto [operand, input] : llvm::zip(operands, fnType.getInputs()))
    *operand = input;

  // Set result type
  if (fnType.getResults().size() != 1)
    return parser.emitError(loc, "expected single output");
  result = fnType.getResults()[0];

  return success();
}

}  // namespace

namespace detail {

void printSameOperandsAndResultTypeImpl(OpAsmPrinter& p, Operation* op,
                                        TypeRange operands, Type result) {
  // Handle zero operand types `() -> a` prints `a`
  if (operands.empty()) {
    p.printType(result);
    return;
  }

  // Handle all same type `(a,a,...) -> a` prints `a`
  bool allSameType =
      llvm::all_of(operands, [&result](auto t) { return t == result; });
  if (allSameType) {
    p.printType(result);
    return;
  }

  // Fall back to generic
  p.printFunctionalType(op);
}

ParseResult parseSameOperandsAndResultTypeImpl(OpAsmParser& parser,
                                               ArrayRef<Type*> operands,
                                               Type& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  Type type;
  if (parser.parseType(type)) return failure();

  // Handle if function type, all operand types did not match result type.
  if (auto fnType = dyn_cast<FunctionType>(type))
    return assignFromFunctionType(parser, loc, operands, result, fnType);

  // Handle bare types. ` : type` indicating all input/output types match.
  for (Type* t : operands) *t = type;
  result = type;
  return success();
}

}  // namespace detail

void printVariadicSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                            OperandRange operands,
                                            TypeRange opTypes, Type result) {
  return detail::printSameOperandsAndResultTypeImpl(p, op, opTypes, result);
}

ParseResult parseVariadicSameOperandsAndResultType(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
    SmallVectorImpl<Type>& opTypes, Type& result) {
  // Insert a type for each operand. Need to do this since passing the type of
  // a variadic op gives no indication of how many operands were provided.
  opTypes.resize(operands.size());

  // Make a pointer list to the operands
  SmallVector<Type*> typePtrs;
  typePtrs.reserve(opTypes.size());
  for (Type& t : opTypes) typePtrs.push_back(&t);

  return detail::parseSameOperandsAndResultTypeImpl(parser, typePtrs, result);
}

void printConstantOp(OpAsmPrinter& p, Operation* op, ElementsAttr value) {
  assert(op->getNumResults() == 1);
  // If not all types are the same, use generic form.
  if (value.getType() != op->getResultTypes().front()) {
    p.printGenericOp(op, /*printOpName=*/false);
    return;
  }

  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  p << ' ';
  p.printStrippedAttrOrType(value);
}

ParseResult parseConstantOp(OpAsmParser& parser, OperationState& result) {
  // Parse the generic form.
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseRParen())) return failure();
    // Parse optional properties
    if (succeeded(parser.parseOptionalLess()) &&
        (failed(parser.parseAttribute(result.propertiesAttr)) ||
         failed(parser.parseGreater())))
      return failure();

    // Parse optional attributes
    if (failed(parser.parseOptionalAttrDict(result.attributes)))
      return failure();

    // Parse type signature
    if (failed(parser.parseColon()) || failed(parser.parseLParen()) ||
        failed(parser.parseRParen()) || failed(parser.parseArrow()))
      return failure();
    Type resultTy;
    if (failed(parser.parseType(resultTy))) return failure();
    result.addTypes(resultTy);
    return success();
  }

  ElementsAttr valueAttr;
  if (failed(parser.parseOptionalAttrDict(result.attributes))) return failure();
  if (failed(parser.parseCustomAttributeWithFallback(valueAttr, Type{}, "value",
                                                     result.attributes)))
    return failure();
  result.addTypes(valueAttr.getType());
  return success();
}

void printTupleOpType(OpAsmPrinter& p, Operation*, TypeRange operands,
                      Type result) {
  p.printType(result);
}

ParseResult parseTupleOpType(OpAsmParser& parser,
                             SmallVectorImpl<Type>& operands, Type& result) {
  // Result type must be tuple type.
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseType(result))) return failure();

  auto tupType = dyn_cast<TupleType>(result);
  if (!tupType) return parser.emitError(loc, "expected tuple type");

  // Assign operand types to tuple types
  llvm::append_range(operands, tupType.getTypes());
  return success();
}

void printPairwiseOpType(OpAsmPrinter& p, Operation*, TypeRange operands,
                         TypeRange results) {
  llvm::interleaveComma(operands, p);
}

ParseResult parsePairwiseOpType(OpAsmParser& parser,
                                SmallVectorImpl<Type>& operands,
                                SmallVectorImpl<Type>& results) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (failed(parser.parseTypeList(operands)))
    return parser.emitError(loc, "expected type list");
  results = operands;
  return success();
}

void printVariadicOperandWithAttribute(OpAsmPrinter& p, Operation*,
                                       OperandRange operands) {
  llvm::interleaveComma(operands, p);
  p << ",";
}

ParseResult parseVariadicOperandWithAttribute(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) {
  // Parse operands as well as trailing commas. Stops when first non-ssa value
  // seen.
  OpAsmParser::UnresolvedOperand operand;
  OptionalParseResult resultOpt = parser.parseOptionalOperand(operand);
  while (resultOpt.has_value() && succeeded(resultOpt.value())) {
    operands.push_back(operand);
    if (failed(parser.parseComma())) return failure();
    resultOpt = parser.parseOptionalOperand(operand);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Operation Printers and Parsers
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Attribute Printers and Parsers
//===----------------------------------------------------------------------===//

void printSliceRanges(OpAsmPrinter& p, Operation* op,
                      ArrayRef<int64_t> startIndices,
                      ArrayRef<int64_t> limitIndices,
                      ArrayRef<int64_t> strides) {
  p << "[";
  // Let's be safe if we're printing invalid IR somehow: this can't be parsed
  // back!
  if (startIndices.size() != limitIndices.size() ||
      startIndices.size() != strides.size()) {
    p << "start_indices: ";
    llvm::interleaveComma(startIndices, p);
    p << ", limit_indices: ";
    llvm::interleaveComma(limitIndices, p);
    p << ", strides: ";
    llvm::interleaveComma(strides, p);
    p << "]";
    return;
  }

  llvm::interleaveComma(llvm::zip(startIndices, limitIndices, strides), p,
                        [&](std::tuple<int64_t, int64_t, int64_t> pack) {
                          auto [start, limit, stride] = pack;
                          p << start << ":" << limit;
                          if (stride != 1) {
                            p << ":" << stride;
                          }
                        });
  p << "]";
}

ParseResult parseSliceRanges(OpAsmParser& parser,
                             DenseI64ArrayAttr& startIndices,
                             DenseI64ArrayAttr& limitIndices,
                             DenseI64ArrayAttr& strides) {
  if (parser.parseLSquare()) return failure();
  // Parse groups of comma-separated: `start`:`limit`[:`stride`]
  // If the stride isn't provided it'll be 1.
  SmallVector<int64_t> start, limit, stride;
  if (failed(parser.parseOptionalRSquare())) {
    do {
      start.emplace_back();
      limit.emplace_back();
      if (failed(parser.parseInteger(start.back())) ||
          failed(parser.parseColon()) ||
          failed(parser.parseInteger(limit.back())))
        return failure();
      if (failed(parser.parseOptionalColon())) {
        stride.push_back(1);
      } else {
        stride.emplace_back();
        if (failed(parser.parseInteger(stride.back()))) return failure();
      }
      if (succeeded(parser.parseOptionalRSquare())) break;
      if (failed(parser.parseComma())) return failure();
    } while (true);
  }

  startIndices = parser.getBuilder().getDenseI64ArrayAttr(start);
  limitIndices = parser.getBuilder().getDenseI64ArrayAttr(limit);
  strides = parser.getBuilder().getDenseI64ArrayAttr(stride);

  return success();
}

ParseResult dimSizeFromString(AsmParser& parser, int64_t& result) {
  if (succeeded(parser.parseOptionalQuestion())) {
    result = ShapedType::kDynamic;
    return success();
  }
  return parser.parseInteger(result);
}

std::string dimSizeToString(int64_t dimSize) {
  if (hlo::isDynamicDimSize(dimSize)) return "?";
  return std::to_string(dimSize);
}

template <typename Stream>
void printDimSizes(Stream& stream, ArrayRef<int64_t> dimSizes) {
  stream << '[';
  llvm::interleaveComma(dimSizes, stream, [&](int64_t dimSize) {
    stream << dimSizeToString(dimSize);
  });
  stream << ']';
}

std::string dimSizesToString(ArrayRef<int64_t> dimSizes) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  printDimSizes(os, dimSizes);
  return buffer;
}

void printDimSizes(AsmPrinter& p, ArrayRef<int64_t> dimSizes) {
  printDimSizes<AsmPrinter>(p, dimSizes);
}

FailureOr<SmallVector<int64_t>> parseDimSizes(AsmParser& parser) {
  SmallVector<int64_t> dimSizes;
  if (failed(parseDimSizes(parser, dimSizes))) return failure();
  return dimSizes;
}

ParseResult parseDimSizes(AsmParser& parser, SmallVector<int64_t>& dimSizes) {
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
    return dimSizeFromString(parser, dimSizes.emplace_back());
  });
}

}  // namespace mlir::hlo
