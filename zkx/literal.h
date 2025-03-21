/* Copyright 2016 The OpenXLA Authors.

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

#ifndef ZKX_LITERAL_H_
#define ZKX_LITERAL_H_

#include <stddef.h>

#include <algorithm>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"

#include "zkx/index_util.h"
#include "zkx/layout_util.h"
#include "zkx/maybe_owning.h"
#include "zkx/primitive_util.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"
#include "zkx/util.h"

namespace zkx {

class LiteralBase {
 public:
  using DynamicSizeType = ShapeUtil::DynamicSizeType;

  virtual ~LiteralBase() = 0;

  // Literals are equal if they have compatible shapes and the same data
  // values. Layout is not compared. For a layout sensitive comparison
  // call Equal() with layout_sensitive=true.
  bool operator==(const LiteralBase& other) const {
    return Equal(other, false);
  }
  bool operator!=(const LiteralBase& other) const { return !(*this == other); }

  // Compares two literals with optional layout sensitivity. If you use
  // literals in a hash map, together with AbslHashValue or Hash defined below,
  // you must use this method instead of operator== to ensure proper layout
  // handling.
  bool Equal(const LiteralBase& other, bool layout_sensitive) const;

  // Returns the shape of the literal.
  const Shape& shape() const;

  // Returns a Span of the array for this literal for the given NativeT
  // (e.g., float). CHECKs if the subshape of the literal at the given
  // ShapeIndex is not array. See primitive_util.h for the mapping from ZKX type
  // to native type.
  template <typename NativeT>
  absl::Span<const NativeT> data(const ShapeIndex& shape_index = {}) const;

  // Returns a const pointer to (or size of) the underlying buffer holding the
  // array at the given shape index. CHECKs if the subshape of the literal at
  // the given ShapeIndex is not array.
  const void* untyped_data(const ShapeIndex& shape_index = {}) const;
  int64_t size_bytes(const ShapeIndex& shape_index = {}) const;

  // Get the dynamic size on dim_index in the literal at the given shape_index.
  DynamicSizeType GetDynamicSize(int64_t dim_index,
                                 const ShapeIndex& shape_index) const;
  DynamicSizeType GetDynamicSize(int64_t dim_index) const;

  // Returns the count of the elements in the array at the given shape index in
  // this literal.
  int64_t element_count(const ShapeIndex& index = {}) const {
    if (index.empty()) {
      // Common case, avoid GetSubshape().
      return ShapeUtil::ElementsIn(shape());
    }
    return ShapeUtil::ElementsIn(ShapeUtil::GetSubshape(shape(), index));
  }

  // Array literals could be in one of the following three states:
  //   1) Known: we have evaluated and known the value of the array literal.
  //   2) Unknown: we have tried to evaluate the array literal, but its value
  //               cannot be evaluated statically.
  //   3) Undetermined: we haven't tried to evaluate the array literal.
  //  Unknown and Undetermined states are only meant to be used within
  //  HloEvaluator. The rest of ZKX assumes array literals are all known.
  //  Literals that are unknown or undetermined can be copied from, using
  //  CopyFrom and Clone, or moved from using move constructor. Accessing values
  //  of such literals causes undefined behavior.
  enum class ArrayValueState { kKnown = 0, kUnknown = 1, kUndetermined = 2 };

  class Piece {
   public:
    ArrayValueState get_array_value_state() const { return array_value_state_; }
    void set_array_value_state(ArrayValueState state) {
      array_value_state_ = state;
    }
    // Returns the buffer holding the array data for this piece as an array
    // slice. This piece must be array-shaped.
    template <typename NativeT>
    absl::Span<const NativeT> data() const;
    template <typename NativeT>
    absl::Span<NativeT> data();

    // Returns the buffer holding the array data for this piece as a void*. This
    // piece must be array-shaped.
    void* untyped_data();
    const void* untyped_data() const;

    // Gets or sets an element in the array at the given index. The multi_index
    // is CHECKed against the dimension sizes of the array.  This piece must be
    // array-shaped.
    template <typename NativeT>
    NativeT Get(absl::Span<const int64_t> index) const;
    template <typename NativeT>
    void Set(absl::Span<const int64_t> index, NativeT value);

    DynamicSizeType GetDynamicSize(int64_t dim_index) const;
    void SetDynamicSize(int64_t dim_index, DynamicSizeType size);
    void AllocateBuffers();
    void DeallocateBuffers();
    // Gets/sets the buffer holding the array data.
    const char* buffer() const;
    char* buffer() {
      return const_cast<char*>(const_cast<const Piece*>(this)->buffer());
    }
    void set_buffer(char* buffer) {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      auto* dense_rep = std::holds_alternative<Uninitialized>(rep_)
                            ? &rep_.emplace<DenseRep>()
                            : GetDenseRep();
      DCHECK(dense_rep);
      dense_rep->data = buffer;
    }

    // Gets/sets the buffer holding dynamic sizes.
    const DynamicSizeType* dynamic_size_buffer() const {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      return reinterpret_cast<const DynamicSizeType*>(
          buffer() + dynamic_size_buffer_offset());
    }
    DynamicSizeType* dynamic_size_buffer() {
      return const_cast<DynamicSizeType*>(
          const_cast<const Piece*>(this)->dynamic_size_buffer());
    }

    int64_t dynamic_size_buffer_bytes() const {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      return subshape().dimensions_size() * sizeof(DynamicSizeType);
    }

    // Gets or sets the subshape of this piece. This reference points to a
    // subshape within the shape in the containing Literal (Literal::shape_).
    const Shape& subshape() const { return *subshape_; }
    void set_subshape(const Shape* subshape) {
      subshape_ = subshape;
      if (std::holds_alternative<Uninitialized>(rep_)) {
        if (subshape_->IsTuple()) {
          rep_.emplace<TupleRep>();
        }
      }
    }

    // Returns the size in bytes of the buffer holding the dense array data.
    int64_t size_bytes_dense() const {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      return ShapeUtil::ByteSizeOf(subshape());
    }

    // The dynamic metadata starts at the end of the data in the literal.
    // The literal can have any number of bytes. For example, it could be a PRED
    // with 7 elements. `dynamic_size_buffer_offset` returns the number of bytes
    // before the dynamic size information including whatever padding is needed
    // to align the start of the dynamic size information so that it is aligned
    // to a multiple of `sizeof(DynamicSizeType)`.
    int64_t dynamic_size_buffer_offset() const {
      // Make sure the dynamic buffer starts on a boundary aligned to
      // `sizeof(DynamicSizeType)`.
      return RoundUpTo<int64_t>(size_bytes_dense(), sizeof(DynamicSizeType));
    }

    // Total size in bytes, including the dynamic size addition.
    //
    // The shape can become dynamic after this literal is allocated, so we
    // over-allocate the margin for the dynamic shape description in case we
    // need it.
    int64_t total_bytes_dense() const {
      return dynamic_size_buffer_offset() + dynamic_size_buffer_bytes();
    }

    // Returns the number of elements in this piece's array.
    int64_t element_count() const { return ShapeUtil::ElementsIn(subshape()); }

    // Returns the child piece at 'index' of this piece.
    Piece& child(int64_t index) {
      return const_cast<Piece&>(const_cast<const Piece*>(this)->child(index));
    }
    const Piece& child(int64_t index) const {
      auto* tuple_rep = GetTupleRep();
      DCHECK(tuple_rep);
      return tuple_rep->children[index];
    }

    // Adds a child piece to this piece's children.
    void emplace_back(Piece child_piece) {
      auto* tuple_rep = GetTupleRep();
      DCHECK(tuple_rep);
      tuple_rep->children.emplace_back(std::move(child_piece));
    }

    // Returns the size of children pieces of this piece.
    int64_t children_size() const {
      if (auto* tuple_rep = GetTupleRep()) {
        return tuple_rep->children.size();
      }
      return 0;
    }

    // Visitor functions that recursively traverses the piece and calls the
    // given function at each child piece. The function has the type:
    //    void (const ShapeIndex& index, const Piece& piece)
    template <typename Fn>
    void ForEachSubpiece(const Fn& func) const {
      ShapeIndex index;
      return ForEachHelper(
                 [&func](const ShapeIndex& index, const Piece& piece) {
                   func(index, piece);
                   return absl::OkStatus();
                 },
                 *this, &index)
          .IgnoreError();
    }

    // Same as above, but the function has the type:
    //    absl::Status (const ShapeIndex& index, const Piece& piece)
    // The first non-OK return value is returned by the function.
    template <typename Fn>
    absl::Status ForEachSubpieceWithStatus(const Fn& func) const {
      ShapeIndex index;
      return ForEachHelper(func, *this, &index);
    }

    // Same as above, but the function has the type:
    //    Bool (const ShapeIndex& index, const Piece& piece)
    // The first non-true return value is returned by the function.
    template <typename Fn>
    bool ForEachSubpieceWithBool(const Fn& func) const {
      ShapeIndex index;
      return ForEachHelperBool(func, *this, &index);
    }

    // Same as above, but the function has the type:
    //    Void (const ShapeIndex& index, Piece& piece)
    template <typename Fn>
    void ForEachMutableSubpiece(const Fn& func) {
      ShapeIndex index;
      return ForEachMutableHelper(
                 [&func](const ShapeIndex& index, Piece* piece) {
                   func(index, piece);
                   return absl::OkStatus();
                 },
                 const_cast<LiteralBase::Piece*>(this), &index)
          .IgnoreError();
    }
    // Same as above, but the function has the type:
    //    absl::Status (const ShapeIndex& index, Piece& piece)
    // The first non-OK return value is returned by the function.
    template <typename Fn>
    absl::Status ForEachMutableSubpieceWithStatus(const Fn& func) {
      ShapeIndex index;
      return ForEachMutableHelper(func, const_cast<LiteralBase::Piece*>(this),
                                  &index);
    }

    // Returns true if this piece and 'other' contain the same data. This piece
    // and 'other' must be array-shaped and compatible. If a literal has dynamic
    // shape, comparison is done only for the valid elements.
    bool EqualElements(const Piece& other) const;

   private:
    // Uninitialized state representation.
    struct Uninitialized {};
    // Out of line dense array storage.
    union DenseRep {
      char* data;
    };
    struct TupleRep {
      // Children pieces for tuple shaped pieces.
      std::vector<Piece> children = {};
    };

    // Literals can be used as DMA targets, which can require alignment. We
    // force a tsl::Allocator::kAllocatorAlignment-byte minimum
    // alignment.
    static inline constexpr size_t kMinimumAlignment = 64;

    // Use just so many bytes that we don't increase the sizeof(Piece).
    static inline constexpr size_t kMaxInlinedBytes =
        std::max(sizeof(DenseRep), sizeof(TupleRep));

    // Inlined dense array storage.
    struct DenseInlinedRep {
      alignas(kMinimumAlignment) char data[kMaxInlinedBytes];
    };

    const DenseInlinedRep* GetDenseInlinedRep() const {
      return std::get_if<DenseInlinedRep>(&rep_);
    }
    DenseInlinedRep* GetDenseInlinedRep() {
      return std::get_if<DenseInlinedRep>(&rep_);
    }

    const DenseRep* GetDenseRep() const { return std::get_if<DenseRep>(&rep_); }
    DenseRep* GetDenseRep() { return std::get_if<DenseRep>(&rep_); }

    const TupleRep* GetTupleRep() const { return std::get_if<TupleRep>(&rep_); }
    TupleRep* GetTupleRep() { return std::get_if<TupleRep>(&rep_); }

    // Helpers for traversing the piece via ForEachSubpiece rooted at 'index'.
    // The first non-OK (or non-true) value is returned by the function.
    // The callable 'func' has the same signature as described above in
    // ForEachSubpiece*.
    template <typename Fn>
    absl::Status ForEachHelper(const Fn& func, const Piece& piece,
                               ShapeIndex* index) const {
      TF_RETURN_IF_ERROR(func(*index, piece));
      if (auto* tuple_rep = piece.GetTupleRep()) {
        for (int64_t i = 0; i < tuple_rep->children.size(); ++i) {
          index->push_back(i);
          TF_RETURN_IF_ERROR(
              ForEachHelper(func, tuple_rep->children[i], index));
          index->pop_back();
        }
      }
      return absl::OkStatus();
    }
    template <typename Fn>
    bool ForEachHelperBool(const Fn& func, const Piece& piece,
                           ShapeIndex* index) const {
      if (!func(*index, piece)) {
        return false;
      }
      if (auto* tuple_rep = piece.GetTupleRep()) {
        for (int64_t i = 0; i < tuple_rep->children.size(); ++i) {
          index->push_back(i);
          if (!ForEachHelperBool(func, tuple_rep->children[i], index)) {
            return false;
          }
          index->pop_back();
        }
      }
      return true;
    }
    template <typename Fn>
    absl::Status ForEachMutableHelper(const Fn& func, Piece* piece,
                                      ShapeIndex* index) {
      TF_RETURN_IF_ERROR(func(*index, piece));
      if (auto* tuple_rep = piece->GetTupleRep()) {
        for (int64_t i = 0; i < tuple_rep->children.size(); ++i) {
          index->push_back(i);
          TF_RETURN_IF_ERROR(
              ForEachMutableHelper(func, &tuple_rep->children[i], index));
          index->pop_back();
        }
      }
      return absl::OkStatus();
    }

    // Recursive helper for EqualElements.
    template <typename NativeT>
    bool EqualElementsInternal(const Piece& other,
                               std::vector<int64_t>* multi_index) const;

    // Storage representation of this piece.
    std::variant<Uninitialized, DenseInlinedRep, DenseRep, TupleRep> rep_;

    // The shape of piece. This points into the shape of the containing Literal
    // (Literal::shape_).
    const Shape* subshape_ = nullptr;

    ArrayValueState array_value_state_ = ArrayValueState::kKnown;
  };

  const Piece& piece(const ShapeIndex& shape_index) const;

  // Returns the piece at the root of the shape.
  virtual const Piece& root_piece() const = 0;
};

class MutableLiteralBase : public LiteralBase {
 public:
  ~MutableLiteralBase() override = 0;

  // Returns a Span view of the array for this literal for the
  // given NativeT (e.g., float). CHECKs if the subshape of the literal at the
  // given ShapeIndex is not array. See primitive_util.h for the mapping from
  // ZKX type to native type.
  template <typename NativeT>
  absl::Span<NativeT> data(const ShapeIndex& shape_index = {});
  // Unhide const method from parent class.
  using LiteralBase::data;

  // TODO(b/67651157): Remove this accessor. Literal users should not be able to
  // mutate the shape as this can produce malformed Literals.
  Shape* mutable_shape_do_not_use();

  // Set the dynamic size on dim_index in the literal at the given shape_index.
  void SetDynamicSize(int64_t dim_index, const ShapeIndex& shape_index,
                      DynamicSizeType size);
  void SetDynamicSize(int64_t dim_index, DynamicSizeType size);

  // Returns a pointer to the underlying buffer holding the array at the given
  // shape index. CHECKs if the subshape of the literal at the given ShapeIndex
  // is not array.
  void* untyped_data(const ShapeIndex& shape_index = {});
  // Unhide const method from parent class.
  using LiteralBase::untyped_data;

  template <typename NativeT>
  void PopulateR1(absl::Span<const NativeT> values);

  // Fills this literal with the given value.
  template <typename NativeT>
  void PopulateWithValue(NativeT value);

 protected:
  // Returns the piece at the given ShapeIndex.
  Piece& piece(const ShapeIndex& shape_index) {
    return const_cast<Piece&>(LiteralBase::piece(shape_index));
  }

  Piece& mutable_root_piece() { return const_cast<Piece&>(root_piece()); }

  // The literal may or may not own the storage of the shape. Creating/copying a
  // shape can incur significant overhead which in many case we'd like to avoid,
  // esp. for small literals.
  using MaybeOwningShapePtr = MaybeOwning<Shape>;

  // The parent class borrows this shape.
  MaybeOwningShapePtr shape_;
};

// The underlying buffer and shape is always owned by this class.
class Literal : public MutableLiteralBase {
 public:
  Literal();

  // Create a literal of the given shape. The literal is allocated sufficient
  // memory to hold the shape. Memory is uninitialized.
  explicit Literal(const Shape& shape);
  ~Literal() override;

  // Literals are moveable, but not copyable. To copy a literal use
  // Literal::Clone or Literal::CloneToUnique. This prevents inadvertent copies
  // of literals which can be expensive.
  Literal(const Literal& other) = delete;
  Literal& operator=(const Literal& other) = delete;
  Literal(Literal&& other);
  // 'allocate_arrays' indicates whether to allocate memory for the arrays in
  // the shape. If false, buffer pointers inside of the Literal::Pieces are set
  // to nullptr.
  Literal(const Shape& shape, bool allocate_arrays,
          ArrayValueState leaf_array_value_state = ArrayValueState::kKnown);
  Literal& operator=(Literal&& other);

 private:
  friend class LiteralBase;
  friend class MutableLiteralBase;
  const Piece& root_piece() const override { return root_piece_; };
  // Deallocate the buffers held by this literal.
  void DeallocateBuffers();
  // Sets the shape_ field from a Shape. shape_'s element_size_in_bits field
  // on the layout is always set to 0 since Literals do not support packed
  // subbyte elements.
  void SetShape(const Shape& shape);

  // Recursively sets the subshapes and buffers of all subpieces rooted at
  // 'piece'. If 'allocate_array' is true, memory is allocated for the arrays in
  // the shape.
  void SetPiece(
      const Shape& shape, Piece* piece, bool allocate_arrays,
      ArrayValueState leaf_array_value_state = ArrayValueState::kKnown);

  Piece root_piece_;
};

template <typename NativeT>
absl::Span<const NativeT> LiteralBase::Piece::data() const {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  DCHECK(!subshape().has_layout() ||
         subshape().layout().element_size_in_bits() == 0)
      << __func__
      << " is not supported for layouts with custom bit size: " << subshape();
  DCHECK_EQ(subshape().element_type(),
            primitive_util::NativeToPrimitiveType<NativeT>())
      << "Attempting to access "
      << PrimitiveType_Name(primitive_util::NativeToPrimitiveType<NativeT>())
      << " type, but literal element type is "
      << PrimitiveType_Name(subshape().element_type());
  return absl::Span<const NativeT>(reinterpret_cast<const NativeT*>(buffer()),
                                   element_count());
}

template <typename NativeT>
absl::Span<NativeT> LiteralBase::Piece::data() {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  DCHECK(!subshape().has_layout() ||
         subshape().layout().element_size_in_bits() == 0)
      << __func__
      << " is not supported for layouts with custom bit size: " << subshape();
  DCHECK_EQ(subshape().element_type(),
            primitive_util::NativeToPrimitiveType<NativeT>())
      << "Attempting to access "
      << PrimitiveType_Name(primitive_util::NativeToPrimitiveType<NativeT>())
      << " type, but literal element type is "
      << PrimitiveType_Name(subshape().element_type());
  return absl::Span<NativeT>(reinterpret_cast<NativeT*>(buffer()),
                             element_count());
}

template <typename NativeT>
NativeT LiteralBase::Piece::Get(absl::Span<const int64_t> multi_index) const {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  return data<NativeT>()[IndexUtil::MultidimensionalIndexToLinearIndex(
      subshape(), multi_index)];
}

template <typename NativeT>
void LiteralBase::Piece::Set(absl::Span<const int64_t> multi_index,
                             NativeT value) {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  data<NativeT>()[IndexUtil::MultidimensionalIndexToLinearIndex(
      subshape(), multi_index)] = value;
}

template <typename NativeT>
absl::Span<const NativeT> LiteralBase::data(
    const ShapeIndex& shape_index) const {
  return piece(shape_index).data<NativeT>();
}

template <typename NativeT>
absl::Span<NativeT> MutableLiteralBase::data(const ShapeIndex& shape_index) {
  return piece(shape_index).data<NativeT>();
}

template <typename NativeT>
ABSL_ATTRIBUTE_NOINLINE void MutableLiteralBase::PopulateR1(
    absl::Span<const NativeT> values) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  CHECK_EQ(shape().rank(), 1);
  if (shape().is_static()) {
    CHECK_EQ(ShapeUtil::ElementsIn(shape()), values.size());
  } else {
    CHECK_EQ(GetDynamicSize(0), values.size());
  }
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  auto data_span = data<NativeT>();
  std::copy(values.begin(), values.end(), data_span.begin());
}

template <typename NativeT>
void MutableLiteralBase::PopulateWithValue(NativeT value) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  for (NativeT& element : data<NativeT>()) {
    element = value;
  }
}

}  // namespace zkx

#endif  // ZKX_LITERAL_H_
