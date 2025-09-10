/* Copyright 2024 The OpenXLA Authors.

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
#include "zkx/codegen/emitters/elemental_hlo_to_mlir.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "zkx/codegen/emitters/ir/zkx_ops.h"

namespace zkx::emitters {

using llvm::SmallVector;
using llvm::SmallVectorImpl;
using mlir::ImplicitLocOpBuilder;
using mlir::Value;
using mlir::ValueRange;
using mlir::arith::AndIOp;
using mlir::arith::CmpIOp;
using mlir::arith::CmpIPredicate;
using mlir::arith::ConstantIndexOp;
using mlir::arith::ConstantOp;

SmallVector<Value, 3> ApplyIndexing(const IndexingMap& map_in, ValueRange dims,
                                    ValueRange symbols,
                                    ImplicitLocOpBuilder& b) {
  IndexingMap map = map_in;
  map.ClearConstraints();
  SmallVector<Value, 3> results;
  for (unsigned int i = 0; i < map.GetNumResults(); ++i) {
    SmallVector<Value, 1> result;
    b.createOrFold<ApplyIndexingOp>(result, dims, symbols, map.GetSubMap(i));
    results.append(result);
  }
  return results;
}

namespace {

Value CheckConstraint(Value constrained_value, Interval range,
                      ImplicitLocOpBuilder& b) {
  auto lb = b.create<ConstantOp>(b.getIndexAttr(range.lower));
  if (range.IsPoint()) {
    return b.create<CmpIOp>(CmpIPredicate::eq, constrained_value, lb);
  }
  auto ub = b.create<ConstantOp>(b.getIndexAttr(range.upper));
  return b.create<AndIOp>(
      b.create<CmpIOp>(CmpIPredicate::sge, constrained_value, lb),
      b.create<CmpIOp>(CmpIPredicate::sle, constrained_value, ub));
}

}  // namespace

Value CheckConstraints(const IndexingMap& map, ValueRange dims,
                       ValueRange symbols, ImplicitLocOpBuilder& b) {
  SmallVector<mlir::AffineExpr, 1> expressions;
  for (auto&& [expression, _] : map.GetConstraints()) {
    expressions.push_back(expression);
  }

  // Construct an indexing for the constraints, so we can use `apply_indexing`.
  mlir::AffineMap input_map = map.GetAffineMap();
  IndexingMap constraints_map{
      mlir::AffineMap::get(input_map.getNumDims(), input_map.getNumSymbols(),
                           expressions, input_map.getContext()),
      map.GetDimVars(), map.GetRangeVars(), map.GetRTVars()};
  SmallVector<Value, 1> constraints_values =
      ApplyIndexing(constraints_map, dims, symbols, b);

  Value ret = b.create<ConstantOp>(b.getIntegerAttr(b.getI1Type(), 1));
  for (auto&& [value, expression_and_range] :
       llvm::zip(constraints_values, map.GetConstraints())) {
    ret = b.create<AndIOp>(
        ret, CheckConstraint(value, expression_and_range.second, b));
  }
  for (auto&& [index, bound] : llvm::enumerate(map.GetDimensionBounds())) {
    ret = b.create<AndIOp>(ret, CheckConstraint(dims[index], bound, b));
  }
  return ret;
}

void GetLoopBoundsFromIndexingMap(ImplicitLocOpBuilder& b,
                                  const IndexingMap& indexing_map,
                                  SmallVectorImpl<Value>& lbs,
                                  SmallVectorImpl<Value>& ubs,
                                  SmallVectorImpl<Value>& steps) {
  Value c1 = b.create<ConstantIndexOp>(1);

  for (const Interval& bound : indexing_map.GetSymbolBounds()) {
    lbs.push_back(b.create<ConstantIndexOp>(bound.lower));
    ubs.push_back(b.create<ConstantIndexOp>(bound.upper + 1));
    steps.push_back(c1);
  }
}

}  // namespace zkx::emitters
