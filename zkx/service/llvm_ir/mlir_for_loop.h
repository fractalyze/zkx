/* Copyright 2017 The OpenXLA Authors.

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

#ifndef ZKX_SERVICE_LLVM_IR_MLIR_FOR_LOOP_H_
#define ZKX_SERVICE_LLVM_IR_MLIR_FOR_LOOP_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "zkx/codegen/emitter_loc_op_builder.h"
#include "zkx/service/llvm_ir/mlir_array.h"

namespace zkx::llvm_ir {

enum class UnrollMode {
  kDefaultUnroll,
  kFullyUnroll,
  kNoUnroll,
};

// A class for constructing a for-loop in MLIR IR.
class MlirForLoop {
 public:
  MlirForLoop(const MlirForLoop&) = delete;
  MlirForLoop& operator=(const MlirForLoop&) = delete;

  // Emit a for-loop at the current insert point of the given IRBuilder.
  //
  // `start_index` and `end_index` are the loop bounds (`end_index` is not
  // inclusive). `step` is the increment of the loop index after each iteration.
  //
  // The current insert block of the builder is the preheader to the loop
  // (see below for definition of block names). All instructions (if any)
  // at or after the insert point in the insert block are moved to a newly
  // created exit block. Instructions before the insert point remain in
  // the insert block:
  //
  //                   +--------------+         +-------------------+
  //                   | insert block |         |    insert block   |
  //                   |     ...      |         | (preheader block) |
  //                   | %foo = ...   |         |      ...          |
  //    insert point ->| %bar = ...   |  ===>   | %foo = ...        |
  //                   |     ...      |         +-------------------+
  //                   +--------------+                 |
  //                                                    V
  //                                              [[ LOOP blocks ]]
  //                                                    |
  //                                                    V
  //                                             +--------------+
  //                                             |  exit block  |
  //                                             | %bar = ...   |
  //                                             |     ...      |
  //                                             +--------------+
  //
  // `prefix` is used to disambiguate variable and block names emitted in
  // MLIR IR. If non-empty, it is prepended to the name of the induction
  // variable value and each block created for the loop.
  //
  // `unroll_mode` specifies the desired LLVM unrolling behavior for generated
  //  loop.
  static std::unique_ptr<MlirForLoop> EmitForLoop(
      std::string_view prefix, mlir::Value start_index, mlir::Value end_index,
      mlir::Value step, EmitterLocOpBuilder& b, mlir::Type index_type,
      UnrollMode unroll_mode = UnrollMode::kDefaultUnroll,
      bool prevent_vectorization = false);

  // The names of the blocks follow LLVM's conventions. Control flow amongst the
  // blocks for the example C code looks like:
  //
  //   for (int i = 0; i < n; ++i) {
  //     do_stuff(i);
  //   }
  //
  //      +-----------------+
  //      | preheader block |
  //      |     i = 0       |
  //      +-----------------+
  //              |
  //              V
  //      +--------------+
  //      | header block |<-+
  //      | if i < n:    |  |
  //      |   goto body  |  |
  //      | else:        |  |
  //      |   goto exit  |  |
  //      +--------------+  |
  //            | |         |
  //   +--------+ |         |
  //   |          V         |
  //   |  +-------------+   |
  //   |  |  body block |   |
  //   |  | dostuff(i)  |---+
  //   |  | ++i         |
  //   |  +-------------+
  //   |
  //   |  +-------------+
  //   +->|  exit block |
  //      +-------------+
  //
  // Caller-emitted code to execute within the loop should be placed within the
  // "body" block.
  //
  // Return pointers to various blocks in the loop.
  mlir::Block* GetPreheaderBlock() const { return preheader_block_; }
  mlir::Block* GetHeaderBlock() const { return header_block_; }
  mlir::Block* GetBodyBlock() const { return body_block_; }
  mlir::Block* GetExitBlock() const { return exit_block_; }

  // Return the Value representing the induction variable in the body block of
  // the loop.
  mlir::Value GetIndVarValue() const { return indvar_; }

 private:
  // Allow MlirForLoopNest to call this private constructor.
  friend class MlirForLoopNest;

  MlirForLoop(std::string_view prefix, std::string_view suffix,
              mlir::Value start_index, mlir::Value end_index, mlir::Value step,
              UnrollMode unroll_mode, bool prevent_vectorization);

  // Emit the loop at the insert point of the builder.
  void Emit(EmitterLocOpBuilder& b, mlir::Type index_type);

  mlir::Block* CreateLoopBlock(std::string_view name, EmitterLocOpBuilder& b);

  // Return an annotation that should be associated with this `MlirForLoop`.
  mlir::LLVM::LoopAnnotationAttr GetLoopAttr() const;

  std::string prefix_;
  std::string suffix_;
  mlir::Value start_index_;
  mlir::Value end_index_;
  mlir::Value step_;

  // To improve readability of the IR, we want the basic blocks to appear
  // consecutively in the following order: preheader, header, body, loop,
  // exit. The member `insert_before_block_` points to where the next basic
  // block should be created to ensure this ordering.
  mlir::Block* insert_before_block_;

  mlir::Block* preheader_block_;
  mlir::Block* header_block_;
  mlir::Block* body_block_;
  mlir::Block* exit_block_;
  mlir::Value indvar_;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-private-field"
  UnrollMode unroll_mode_;
  bool prevent_vectorization_;
#pragma GCC diagnostic pop
};

// A simple class for constructing nested for-loops.
class MlirForLoopNest {
 public:
  MlirForLoopNest(std::string_view name, EmitterLocOpBuilder& b,
                  mlir::Type index_ty = nullptr)
      : name_(name),
        outer_loop_preheader_block_(nullptr),
        outer_loop_exit_block_(nullptr),
        inner_loop_body_block_(nullptr),
        b_(b) {
    SetIndexType(index_ty);
  }
  MlirForLoopNest(const MlirForLoopNest&) = delete;
  MlirForLoopNest& operator=(const MlirForLoopNest&) = delete;

  // Adds a loop to the nest. If no loop has been added yet then emit a loop at
  // the current insert point of the given builder. If one or more loops have
  // been added then emit loop inside the body of the last added loop.
  // `unroll_mode` is used to emit metadata that controls LLVM unrolling.
  std::unique_ptr<MlirForLoop> AddLoop(
      std::string_view suffix, mlir::Value start_index, mlir::Value end_index,
      mlir::Value stride, UnrollMode unroll_mode = UnrollMode::kDefaultUnroll,
      bool prevent_vectorization = false);

  // Like the above, except that it defaults to a stride of one.
  std::unique_ptr<MlirForLoop> AddLoop(
      std::string_view suffix, mlir::Value start_index, mlir::Value end_index,
      UnrollMode unroll_mode = UnrollMode::kDefaultUnroll,
      bool prevent_vectorization = false);

  // A convenient wrapper of the other flavor of AddLoop. The given start and
  // end index are constant.
  std::unique_ptr<MlirForLoop> AddLoop(
      int64_t start_index, int64_t end_index, int64_t stride,
      std::string_view suffix,
      UnrollMode unroll_mode = UnrollMode::kDefaultUnroll,
      bool prevent_vectorization = false);

  // Like the above, except that it defaults to a stride of one.
  std::unique_ptr<MlirForLoop> AddLoop(
      int64_t start_index, int64_t end_index, std::string_view suffix,
      UnrollMode unroll_mode = UnrollMode::kDefaultUnroll,
      bool prevent_vectorization = false);

  // Add loops to iterate through the indices within the specified
  // shape. The returned index collects the induction variables of the
  // loops so that it will iterate through all coordinates within the
  // specified shape.
  //
  // E.g. if you pass in a 2x3 shape, you will get back an index with
  // two entries that are induction variables of the two loops that
  // will be added. That index will iterate through the 6 coordinates
  // within the shape. One possible order for that sequence would be:
  //
  //   (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
  MlirArray::Index AddLoopsForShape(const Shape& shape,
                                    std::string_view suffix);

  // Add a loop for each dimension in "dimensions". "suffix" is the
  // name suffix of the indvar and basic blocks in this new loop nest.
  //
  // The return value is an index with the induction variables. The
  // size equals the rank of shape and there is a null for each
  // dimension that is not in "dimensions".
  std::vector<mlir::Value> AddLoopsForShapeOnDimensions(
      const Shape& shape, absl::Span<const int64_t> dimensions,
      std::string_view suffix);

  // Emits a series of nested loops for iterating over an operand array. Loops
  // are constructed in major to minor dimension layout order. No loop is
  // emitted for the given `dimension_to_skip`. The function returns an
  // MlirArray index for the given operand_array containing the indvars of the
  // loops. All dimensions of the index are filled except for
  // `dimension_to_skip`. name_suffix is the string to append to the names of
  // LLVM constructs (eg, basic blocks) constructed by this method.
  std::vector<mlir::Value> EmitOperandArrayLoopNest(
      const MlirArray& operand_array, int64_t dimension_to_skip,
      std::string_view name_suffix);

  // Convenience methods which return particular basic blocks of the outermost
  // or innermost loops. These methods return nullptr if no loops have been
  // added yet.
  mlir::Block* GetOuterLoopPreheaderBlock() {
    return outer_loop_preheader_block_;
  }
  mlir::Block* GetOuterLoopExitBlock() { return outer_loop_exit_block_; }
  mlir::Block* GetInnerLoopBodyBlock() { return inner_loop_body_block_; }

 private:
  void SetIndexType(mlir::Type index_ty) {
    index_type_ = index_ty == nullptr ? b_.getI64Type() : index_ty;
  }

  mlir::Value GetConstantWithIndexType(int64_t c) const {
    return b_.create<mlir::LLVM::ConstantOp>(
        index_type_, mlir::IntegerAttr::get(index_type_, c));
  }

  // Human-friendly name of the loop nest.
  std::string name_;

  // The preheader and exit basic block of the outermost loop, or nullptr if no
  // loop has been added yet.
  mlir::Block* outer_loop_preheader_block_;
  mlir::Block* outer_loop_exit_block_;

  // The body basic block of the most-recently added loop, or nullptr if no loop
  // has been added yet.
  mlir::Block* inner_loop_body_block_;

  EmitterLocOpBuilder& b_;

  mlir::Type index_type_;
};

}  // namespace zkx::llvm_ir

#endif  // ZKX_SERVICE_LLVM_IR_MLIR_FOR_LOOP_H_
