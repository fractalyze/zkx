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

#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"

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

}  // namespace zkx

#endif  // ZKX_SHAPE_UTIL_H_
