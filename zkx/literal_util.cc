/* Copyright 2018 The OpenXLA Authors.

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

#include "zkx/literal_util.h"

#include "xla/tsl/platform/status.h"

namespace zkx {

// static
Literal LiteralUtil::MakeTuple(absl::Span<const Literal* const> elements) {
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(elements.size());
  for (const auto* element : elements) {
    element_shapes.push_back(&element->shape());
  }
  Literal literal(ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int i = 0, end = elements.size(); i < end; ++i) {
    TF_CHECK_OK(literal.CopyFrom(*elements[i], /*dest_shape_index=*/{i}));
  }
  return literal;
}

// static
Literal LiteralUtil::MakeTupleFromSlices(
    absl::Span<const LiteralSlice> elements) {
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(elements.size());
  for (const auto& element : elements) {
    element_shapes.push_back(&element.shape());
  }
  Literal literal(ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int i = 0, end = elements.size(); i < end; ++i) {
    TF_CHECK_OK(literal.CopyFrom(elements[i], /*dest_shape_index=*/{i}));
  }
  return literal;
}

// static
Literal LiteralUtil::MakeTupleOwned(std::vector<Literal> elements) {
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(elements.size());
  for (const auto& element : elements) {
    element_shapes.push_back(&element.shape());
  }
  Literal literal(ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int64_t i = 0, end = elements.size(); i < end; ++i) {
    TF_CHECK_OK(
        literal.MoveFrom(std::move(elements[i]), /*dest_shape_index=*/{i}));
  }
  return literal;
}

// static
Literal LiteralUtil::CreateFromDimensions(
    PrimitiveType primitive_type, absl::Span<const int64_t> dimensions) {
  return Literal::CreateFromShape(
      ShapeUtil::MakeShape(primitive_type, dimensions));
}

}  // namespace zkx
