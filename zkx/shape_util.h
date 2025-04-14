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

#ifndef ZKX_SHAPE_UTIL_H_
#define ZKX_SHAPE_UTIL_H_

#include <stdint.h>

#include <ostream>
#include <string>
#include <tuple>

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/overflow_util.h"
#include "zkx/shape.h"

namespace zkx {

// A view into a ShapeIndex below, with the cheap/easy ability to consume the
// value at the front of the view.
//
// NB! ShapeIndexView does not own the memory backing the index array.
// The memory backing the index array should be owned by an object
// that lives longer than the ShapeIndexView instances pointing into
// it.
using ShapeIndexView = absl::Span<const int64_t>;

// TODO(chokobole): add ShapeUtil::GetSubshape
// An index for specifying a particular nested subshape within a shape. Used in
// ShapeUtil::GetSubshape and other interfaces. Shapes are recursive data
// structures (trees) and ShapeIndex defines a path through the tree where each
// element of ShapeIndex indexes into a tuple (or nested tuple) within the
// shape. For a non-nested tuple, an index has a single element. For example,
// given a 3-element tuple (a, b, c) containing arrays a, b, and c, the index
// {1} corresponds to array b. For a nested tuple, the index can have more than
// one element. For the nested tuple (a, (b, c, d), e) below are the values
// corresponding to the given indices:
//
//   index {0}    : array a
//   index {1, 2} : array d
//   index {2}    : array e
//   index {0, 0} : invalid index (element at {0} is an array not a tuple)
//
// For indexing into array shapes, the index is always trivially empty, ie {}.
struct ShapeIndex : public absl::InlinedVector<int64_t, 2> {
  using InlinedVector::InlinedVector;
  ABSL_ATTRIBUTE_NOINLINE ShapeIndex() = default;

  explicit ShapeIndex(ShapeIndexView view)
      : ShapeIndex(view.begin(), view.end()) {}

  // push_front is O(n), but shapes don't usually have a ton of dimensions.
  void push_front(int64_t value) { insert(begin(), value); }
  void pop_front() { erase(begin()); }

  std::string ToString() const;
};

std::ostream& operator<<(std::ostream& out, const ShapeIndex& shape_index);

// Namespaced collection of (static) shape utilities.
//
// These are all effectively convenience functions for testing/tweaking proto
// properties, which do invariant checks before / after the operation.
class ShapeUtil {
 public:
  using DynamicSizeType = int32_t;

  // Returns the product of the statically bound dimensions.
  template <bool kBoundedDynamicOk>
  static inline std::pair<int64_t, bool> ExtentProduct(const Shape& shape) {
    DCHECK(shape.IsArray()) << ShapeUtil::HumanString(shape);
    DCHECK_EQ(shape.dimensions_size(), shape.rank());
    int64_t product = 1;
    bool any_overflows = false;
    for (int dim = 0; dim < shape.dimensions_size(); ++dim) {
      if constexpr (kBoundedDynamicOk) {
        if (shape.is_unbounded_dynamic_dimension(dim)) {
          continue;
        }
      } else {
        DCHECK(!shape.is_unbounded_dynamic_dimension(dim));
      }
      bool overflow;
      std::tie(product, overflow) =
          OverflowSafeMultiply(product, shape.dimensions(dim));
      any_overflows |= overflow;
    }
    return {product, any_overflows};
  }

  // Returns the number of elements contained within the provided shape;
  // e.g. for rank 0 (scalars) the result is always 1.
  // Precondition: shape.IsArray()
  static inline int64_t ElementsIn(const Shape& shape) {
    auto [product, overflow] =
        ExtentProduct</*kBoundedDynamicOk=*/false>(shape);
    DCHECK(!overflow);
    return product;
  }

  // Returns the number of bytes required for an allocation of shape. The
  // |pointer_size| parameter is used for calculating the size of tuple
  // shapes. This includes only the size of the top-level buffer. For example, a
  // tuple is stored as an array of pointers to other buffers. In this case,
  // this method only returns the size of the pointer array.
  static int64_t ByteSizeOf(const Shape& shape, int64_t pointer_size = -1);

  // Returns the number of bytes used to store the primitive_type.
  //
  // Precondition: shape.IsArray()
  static int64_t ByteSizeOfPrimitiveType(PrimitiveType primitive_type);

  // Returns the number of bytes required to store the tuple member pointers for
  // a allocation of shape. The `shape` must be a TUPLE shape, and
  // `pointer_size` must be larger than zero.
  static int64_t ByteSizeOfTupleIndexTable(const Shape& shape,
                                           int64_t pointer_size);

  // Returns the number of bytes required for the elements in an allocation of
  // `shape`, which must be an array shape. Shapes use a separate
  // memory location for each element, and so for these shapes,
  // `ByteSizeOf(shape) == ByteSizeOfElements(shape)`. This
  // size also includes padding if present in the layout.
  static int64_t ByteSizeOfElements(const Shape& shape);

  // Prints a human-readable string that represents the given shape, with or
  // without layout. e.g. "u32[42x12] {0, 1}" or "u32[64]".
  static void PrintHumanString(Printer* printer, const Shape& shape);
  static void PrintHumanStringWithLayout(Printer* printer, const Shape& shape);

  // Returns a human-readable string that represents the given shape, with or
  // without layout. e.g. "u32[42x12] {0, 1}" or "u32[64]".
  static std::string HumanString(const Shape& shape);
  static std::string HumanStringWithLayout(const Shape& shape);

  // Returns whether the LHS and RHS shapes have the same rank; note: does
  // not check element type.
  // Precondition: IsArray(lhs) && IsArray(rhs)
  static bool SameRank(const Shape& lhs, const Shape& rhs) {
    return lhs.rank() == rhs.rank();
  }

  // Returns whether the lhs and rhs shapes have the same element type.
  static bool SameElementType(const Shape& lhs, const Shape& rhs) {
    return lhs.element_type() == rhs.element_type();
  }

  // Returns true if the rank, dimension sizes, and element type are
  // identical. Layout is ignored. Tuple elements are compared recursively for
  // compatibility.
  static bool Compatible(const Shape& lhs, const Shape& rhs);

  // Returns whether the lhs and rhs shapes are identical.
  static bool Equal(const Shape& lhs, const Shape& rhs);

  // Two shapes have same structure if all subshape indices of lhs are presented
  // on rhs and vice versa.
  // A nested tuple shape of (U32, (S32[2], U32[2, 2])) is structurally equal to
  // (S32, (U32[3], S32[2])) as their structures are both (,(,))
  //
  // In contrast, (U32, (U32, U32)) is structurally different from
  // ((U32, U32), U32) as the former has structure (,(,)) while the latter has
  // ((,),)
  static bool EqualStructure(const Shape& lhs, const Shape& rhs);

  ////////////////////
  // Scalar-specific

  static bool IsScalar(const Shape& shape) {
    return shape.IsArray() && shape.rank() == 0;
  }

  // Creates a tuple shape from a slice of element shapes within the tuple.
  static Shape MakeTupleShape(absl::Span<const Shape> shapes);
  static Shape MakeTupleShapeWithPtrs(absl::Span<const Shape* const> shapes);

  // Appends a shape to the given tuple.
  static void AppendShapeToTuple(const Shape& shape, Shape* tuple_shape);

  // Returns an empty tuple shape. Can be used as a sentinel Shape value.
  static Shape MakeNil() { return MakeTupleShape({}); }

  // Constructs a new shape with the given element type and sequence of
  // dimensions.
  static Shape MakeShape(PrimitiveType element_type,
                         absl::Span<const int64_t> dimensions);

  // Make a scalar shape with given primitive type.
  static Shape MakeScalarShape(PrimitiveType element_type);

  // Validates that the provided shape satisfies invariants.
  static absl::Status ValidateShape(const Shape& shape);

  // Validates the provided shape satisfies invariants, except those that
  // pertain to layout.
  //
  // Layout is optional for client-provided shapes, so that the compiler may
  // determine and assign an optimized layout.
  static absl::Status ValidateShapeWithOptionalLayout(const Shape& shape);

  // Returns the number of elements in the given tuple shape.
  // Precondition: IsTuple(shape)
  static int64_t TupleElementCount(const Shape& shape);

  // Returns the number of elements, recursively, in the given shape.
  static int64_t SubshapeCount(const Shape& shape);

  // Returns true if the given shape has a subshape at the given index.
  static bool IndexIsValid(const Shape& shape, ShapeIndexView index);

  // GetSubshape and GetMutableSubshape return a particular nested Shape within
  // the given Shape argument. The non-Try variants check fail if index is
  // invalid.
  static const Shape& GetSubshape(const Shape& shape, ShapeIndexView index);

  // Faster version for one index.
  static const Shape& GetSubshapeOneIndex(const Shape& shape, int64_t index);

  static absl::StatusOr<const Shape*> TryGetSubshape(const Shape& shape,
                                                     ShapeIndexView index);
  static Shape* GetMutableSubshape(Shape* shape, ShapeIndexView index);

  // Calls the given visitor function for each subshape of the given shape.
  // Subshapes are visited in DFS pre-order starting with the entire shape
  // (index {}).
  //
  // The visitor function must have the signature
  //
  //   void fn(const Shape& subshape, const ShapeIndex& index), or
  //   void fn(Shape* subshape, const ShapeIndex& index) (mutable version)
  template <typename Fn>
  static void ForEachSubshape(const Shape& shape, Fn&& fn) {
    ForEachSubshapeWithStatus(shape, [&](const Shape& subshape,
                                         const ShapeIndex& index) {
      fn(subshape, index);
      return absl::OkStatus();
    }).IgnoreError();
  }
  template <typename Fn>
  static void ForEachMutableSubshape(Shape* shape, Fn&& fn) {
    ForEachMutableSubshapeWithStatus(shape, [&](Shape* subshape,
                                                const ShapeIndex& index) {
      fn(subshape, index);
      return absl::OkStatus();
    }).IgnoreError();
  }

  // Variants of ForEach(Mutable)Subshape which propagate absl::Status from the
  // visitor function.
  //
  // Visitor function must have the signature
  //
  //   absl::Status fn(const Shape& subshape, const ShapeIndex& index), or
  //   absl::Status fn(Shape* subshape, const ShapeIndex& index) (mutable
  //   version)
  //
  template <typename Fn>
  static absl::Status ForEachSubshapeWithStatus(const Shape& shape, Fn&& fn) {
    return ForEachMutableSubshapeWithStatus(
        const_cast<Shape*>(&shape),
        [&](Shape* subshape, const ShapeIndex& index) -> absl::Status {
          return fn(*const_cast<const Shape*>(subshape), index);
        });
  }
  template <typename Fn>
  static absl::Status ForEachMutableSubshapeWithStatus(Shape* shape, Fn&& fn) {
    ShapeIndex index;
    return ForEachMutableSubshapeWithStatusHelper(shape, fn, &index);
  }

 private:
  // Fills *shape ignoring dynamic dimensions. Returns true on success.
  // REQUIRES: *shape is empty.
  static bool FillNewShape(PrimitiveType element_type,
                           absl::Span<const int64_t> dimensions, Shape* shape);

  // Helper for ForEachSubshape which visits the subshapes of the given shape in
  // DFS pre-order starting with the index.
  template <typename Fn>
  static absl::Status ForEachMutableSubshapeWithStatusHelper(
      Shape* shape, Fn&& fn, ShapeIndex* index) {
    TF_RETURN_IF_ERROR(fn(shape, *index));
    if (shape->IsTuple()) {
      for (int64_t i = 0; i < TupleElementCount(*shape); ++i) {
        index->push_back(i);
        TF_RETURN_IF_ERROR(ForEachMutableSubshapeWithStatusHelper(
            shape->mutable_tuple_shapes(i), fn, index));
        index->pop_back();
      }
    }
    return absl::OkStatus();
  }
};

}  // namespace zkx

#endif  // ZKX_SHAPE_UTIL_H_
