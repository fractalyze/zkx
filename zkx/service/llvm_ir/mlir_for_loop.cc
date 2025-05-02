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

#include "zkx/service/llvm_ir/mlir_for_loop.h"

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"

#include "zkx/layout_util.h"
#include "zkx/service/llvm_ir/llvm_util.h"

namespace zkx::llvm_ir {

MlirForLoop::MlirForLoop(std::string_view prefix, std::string_view suffix,
                         mlir::Value start_index, mlir::Value end_index,
                         mlir::Value step, UnrollMode unroll_mode,
                         bool prevent_vectorization)
    : prefix_(prefix),
      suffix_(suffix),
      start_index_(start_index),
      end_index_(end_index),
      step_(step),
      insert_before_block_(nullptr),
      unroll_mode_(unroll_mode),
      prevent_vectorization_(prevent_vectorization) {}

// static
std::unique_ptr<MlirForLoop> MlirForLoop::EmitForLoop(
    std::string_view prefix, mlir::Value start_index, mlir::Value end_index,
    mlir::Value step, EmitterLocOpBuilder& b, mlir::Type index_type,
    UnrollMode unroll_mode, bool prevent_vectorization) {
  std::unique_ptr<MlirForLoop> loop(
      new MlirForLoop(prefix, /*suffix=*/"", start_index, end_index, step,
                      unroll_mode, prevent_vectorization));
  loop->Emit(b, index_type);
  return loop;
}

void MlirForLoop::Emit(EmitterLocOpBuilder& b, mlir::Type index_type) {
  // The preheader block is the block the builder is currently emitting
  // code into.
  preheader_block_ = b.getBlock();

  mlir::Block::iterator insertion_point = b.getInsertionPoint();
  if (insertion_point == preheader_block_->end()) {
    // We're emitting the loop at the end of a block.
    exit_block_ = CreateLoopBlock("loop_exit", b);
  } else {
    // We're emitting the loop into the middle of a block. splitBlock
    // requires that this block be well-formed (have a terminator).
    CHECK_NE(nullptr, preheader_block_->getTerminator());

    // Split the preheader to create an exit block. The exit block
    // will contain all instructions at or after `insertion_point`.
    exit_block_ = preheader_block_->splitBlock(insertion_point);
  }
  insert_before_block_ = exit_block_;

  // Create remaining block which form the inside of the loop.
  header_block_ = CreateLoopBlock("loop_header", b);
  body_block_ = CreateLoopBlock("loop_body", b);

  // Function entry block.
  // Emit alloca for the induction variable. We do this at the entry to the
  // block to ensure the alloc only executes once per function (we could
  // be emitting a nested loop).
  mlir::Region* parent_region = preheader_block_->getParent();
  b.setInsertionPointToStart(&parent_region->front());
  mlir::Value indvar_address = b.create<mlir::LLVM::AllocaOp>(
      b.getType<mlir::LLVM::LLVMPointerType>(), start_index_.getType(),
      b.create<mlir::LLVM::ConstantOp>(index_type,
                                       b.getIntegerAttr(index_type, 1)));

  // Preheader block.
  // Initialize induction variable starting index. Create branch to the header.
  b.setInsertionPointToEnd(preheader_block_);

  b.create<mlir::LLVM::StoreOp>(start_index_, indvar_address);
  // The preheader should not have a branch yet.
  b.create<mlir::LLVM::BrOp>(header_block_);

  // Header block.
  // Emit the loop conditional branch. Load and compare indvar with ending
  // index and jump to loop exit if equal. Jump to body otherwise.
  b.setInsertionPointToEnd(header_block_);
  indvar_ =
      b.create<mlir::LLVM::LoadOp>(start_index_.getType(), indvar_address);
  mlir::Value exit_cond = b.create<mlir::LLVM::ICmpOp>(
      b.getI1Type(), mlir::LLVM::ICmpPredicate::uge, indvar_, end_index_);
  b.create<mlir::LLVM::CondBrOp>(exit_cond, exit_block_, body_block_);

  // Body block.
  // Increment indvar, store indvar, and jump to header.
  b.setInsertionPointToEnd(body_block_);
  mlir::Value step = step_;
  mlir::Value indvar = indvar_;

  mlir::Value indvar_inc =
      b.create<mlir::LLVM::AddOp>(indvar.getType(), indvar, step,
                                  mlir::LLVM::IntegerOverflowFlags::nsw |
                                      mlir::LLVM::IntegerOverflowFlags::nuw);
  b.create<mlir::LLVM::StoreOp>(indvar_inc, indvar_address);
  auto back_branch = b.create<mlir::LLVM::BrOp>(header_block_);

  if (unroll_mode_ != UnrollMode::kDefaultUnroll || prevent_vectorization_) {
    back_branch->setAttr("llvm.loop", GetLoopAttr());
  }

  // Re-point the IR builder to the loop exit block.
  b.setInsertionPointToEnd(exit_block_);
}

mlir::LLVM::LoopAnnotationAttr MlirForLoop::GetLoopAttr() const {
  mlir::MLIRContext* context = start_index_.getContext();
  auto true_attr = mlir::BoolAttr::get(context, true);

  mlir::LLVM::LoopUnrollAttr unroll = nullptr;
  if (unroll_mode_ == UnrollMode::kNoUnroll) {
    unroll = mlir::LLVM::LoopUnrollAttr::get(context,
                                             /*disable=*/true_attr,
                                             /*count=*/nullptr,
                                             /*runtimeDisable=*/nullptr,
                                             /*full=*/nullptr,
                                             /*followupUnrolled=*/nullptr,
                                             /*followupRemainder=*/nullptr,
                                             /*followupAll=*/nullptr);
  } else if (unroll_mode_ == UnrollMode::kFullyUnroll) {
    unroll = mlir::LLVM::LoopUnrollAttr::get(context,
                                             /*disable=*/nullptr,
                                             /*count=*/nullptr,
                                             /*runtimeDisable=*/nullptr,
                                             /*full=*/true_attr,
                                             /*followupUnrolled=*/nullptr,
                                             /*followupRemainder=*/nullptr,
                                             /*followupAll=*/nullptr);
  }

  mlir::LLVM::LoopVectorizeAttr vectorize = nullptr;
  if (prevent_vectorization_) {
    vectorize = mlir::LLVM::LoopVectorizeAttr::get(
        context,
        /*disable=*/true_attr, /*predicateEnable=*/nullptr,
        /*scalableEnable=*/nullptr, /*width=*/nullptr,
        /*followupVectorized=*/nullptr, /*followupEpilogue=*/nullptr,
        /*followupAll=*/nullptr);
  }

  return mlir::LLVM::LoopAnnotationAttr::get(context,
                                             /*disableNonforced=*/nullptr,
                                             /*vectorize=*/vectorize,
                                             /*interleave=*/nullptr,
                                             /*unroll=*/unroll,
                                             /*unrollAndJam=*/nullptr,
                                             /*licm=*/nullptr,
                                             /*distribute=*/nullptr,
                                             /*pipeline=*/nullptr,
                                             /*peeled=*/nullptr,
                                             /*unswitch=*/nullptr,
                                             /*mustProgress=*/nullptr,
                                             /*isVectorized=*/nullptr,
                                             /*startLoc=*/nullptr,
                                             /*endLoc=*/nullptr,
                                             /*parallelAccesses=*/{});
}

mlir::Block* MlirForLoop::CreateLoopBlock(std::string_view name,
                                          EmitterLocOpBuilder& b) {
  if (insert_before_block_) {
    mlir::Region* parent_region = insert_before_block_->getParent();
    mlir::Block* new_block = new mlir::Block();
    parent_region->getBlocks().insert(
        mlir::Region::iterator(insert_before_block_), new_block);
    return new_block;
  } else {
    mlir::Region* parent_region = preheader_block_->getParent();
    return &parent_region->emplaceBlock();
  }
}

std::unique_ptr<MlirForLoop> MlirForLoopNest::AddLoop(
    std::string_view suffix, mlir::Value start_index, mlir::Value end_index,
    UnrollMode unroll_mode, bool prevent_vectorization) {
  return AddLoop(suffix, start_index, end_index, GetConstantWithIndexType(1),
                 unroll_mode, prevent_vectorization);
}

std::unique_ptr<MlirForLoop> MlirForLoopNest::AddLoop(
    std::string_view suffix, mlir::Value start_index, mlir::Value end_index,
    mlir::Value stride, UnrollMode unroll_mode, bool prevent_vectorization) {
  std::unique_ptr<MlirForLoop> loop(new MlirForLoop(
      /*prefix=*/name_, suffix, start_index, end_index, stride, unroll_mode,
      prevent_vectorization));
  loop->Emit(b_, index_type_);

  if (outer_loop_preheader_block_ == nullptr) {
    outer_loop_preheader_block_ = loop->GetPreheaderBlock();
  }

  if (outer_loop_exit_block_ == nullptr) {
    outer_loop_exit_block_ = loop->GetExitBlock();
  }

  inner_loop_body_block_ = loop->GetBodyBlock();

  return loop;
}

std::unique_ptr<MlirForLoop> MlirForLoopNest::AddLoop(
    int64_t start_index, int64_t end_index, std::string_view suffix,
    UnrollMode unroll_mode, bool prevent_vectorization) {
  CHECK_LE(start_index, end_index);
  return AddLoop(suffix, GetConstantWithIndexType(start_index),
                 GetConstantWithIndexType(end_index), unroll_mode,
                 prevent_vectorization);
}

std::unique_ptr<MlirForLoop> MlirForLoopNest::AddLoop(
    int64_t start_index, int64_t end_index, int64_t stride,
    std::string_view suffix, UnrollMode unroll_mode,
    bool prevent_vectorization) {
  CHECK_LE(start_index, end_index);
  return AddLoop(suffix, GetConstantWithIndexType(start_index),
                 GetConstantWithIndexType(end_index),
                 GetConstantWithIndexType(stride), unroll_mode,
                 prevent_vectorization);
}

MlirArray::Index MlirForLoopNest::AddLoopsForShape(const Shape& shape,
                                                   std::string_view suffix) {
  std::vector<int64_t> dimensions(shape.rank());
  std::iota(dimensions.begin(), dimensions.end(), 0);
  return MlirArray::Index(
      AddLoopsForShapeOnDimensions(shape, dimensions, suffix), shape,
      index_type_);
}

std::vector<mlir::Value> MlirForLoopNest::AddLoopsForShapeOnDimensions(
    const Shape& shape, absl::Span<const int64_t> dimensions,
    std::string_view suffix) {
  std::vector<mlir::Value> multi_index(shape.dimensions_size());
  for (int64_t dimension : dimensions) {
    std::unique_ptr<MlirForLoop> loop = AddLoop(
        /*start_index=*/0,
        /*end_index=*/shape.dimensions(dimension),
        /*suffix=*/
        IrName(suffix, absl::StrCat(dimension)));
    multi_index[dimension] = loop->GetIndVarValue();
  }
  return multi_index;
}

std::vector<mlir::Value> MlirForLoopNest::EmitOperandArrayLoopNest(
    const MlirArray& operand_array, int64_t dimension_to_skip,
    std::string_view name_suffix) {
  // Prepares the dimension list we will use to emit the loop nest. Outermost
  // loops are added first. Add loops in major-to-minor order, and skip the
  // `dimension_to_skip` dimension.
  std::vector<int64_t> dimensions;
  const Shape& shape = operand_array.GetShape();
  // Initially get the dimensions in minor to major order, then reverse them.
  for (int64_t dimension : LayoutUtil::MinorToMajor(shape)) {
    if (dimension != dimension_to_skip) {
      dimensions.push_back(dimension);
    }
  }
  absl::c_reverse(dimensions);

  // Create loop nest with one for-loop for each dimension of the
  // output.
  std::vector<mlir::Value> multi_index =
      AddLoopsForShapeOnDimensions(shape, dimensions, name_suffix);
  // Verify every dimension except the `dimension_to_skip` dimension was set in
  // the index.
  for (size_t dimension = 0; dimension < multi_index.size(); ++dimension) {
    if (dimension == dimension_to_skip) {
      DCHECK(!multi_index[dimension]);
    } else {
      DCHECK(multi_index[dimension]);
    }
  }
  return multi_index;
}

}  // namespace zkx::llvm_ir
