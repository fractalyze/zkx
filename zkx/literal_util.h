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

#ifndef ZKX_LITERAL_UTIL_H_
#define ZKX_LITERAL_UTIL_H_

#include "absl/types/span.h"

#include "zkx/literal.h"
#include "zkx/primitive_util.h"
#include "zkx/shape_util.h"

namespace zkx {

class LiteralUtil {
 public:
  template <typename NativeT>
  static Literal CreateR1(absl::Span<const NativeT> values);

  template <typename NativeT>
  static Literal CreateFull(absl::Span<const int64_t> dimensions,
                            NativeT value);
};

// static
template <typename NativeT>
Literal LiteralUtil::CreateR1(absl::Span<const NativeT> values) {
  Literal literal(
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<NativeT>(),
                           {static_cast<int64_t>(values.size())}));
  literal.PopulateR1(values);
  return literal;
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateFull(absl::Span<const int64_t> dimensions,
                                NativeT value) {
  Literal literal(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), dimensions));
  literal.PopulateWithValue(value);
  return literal;
}

}  // namespace zkx

#endif  // ZKX_LITERAL_UTIL_H_
