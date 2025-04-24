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

#include <array>
#include <initializer_list>
#include <iterator>
#include <vector>

#include "absl/types/span.h"

#include "zkx/literal.h"
#include "zkx/primitive_util.h"
#include "zkx/shape_util.h"

namespace zkx {

class LiteralUtil {
 public:
  // Creates a new literal of a given rank. To minimize ambiguity (for users
  // and the compiler) these CreateR[0-2] methods should explicitly specify the
  // native type. For example:
  //
  //  CreateR1<float>({1.0, 42.0});
  //  CreateR2<uint32_t>({{1, 2}, {3, 4}});
  //
  // The variants not ending with WithLayout use the default ZKX layout for the
  // literal's linear representation in memory.
  template <typename NativeT>
  static Literal CreateR0(NativeT value);
  template <typename T>
  static Literal CreateR0(PrimitiveType primitive_type, T value);
  template <typename NativeT>
  static Literal CreateR1(absl::Span<const NativeT> values);
  // TODO(chokobole): Uncomment this. Dependency: Bitmap
  // static Literal CreateR1(const tsl::core::Bitmap& values);
  template <typename NativeT>
  static Literal CreateR2(
      std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  static Literal CreateR2WithLayout(
      std::initializer_list<std::initializer_list<NativeT>> values,
      const Layout& layout);
  template <typename NativeT>
  static Literal CreateR3(std::initializer_list<
                          std::initializer_list<std::initializer_list<NativeT>>>
                              values);
  template <typename NativeT>
  static Literal CreateR3WithLayout(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          values,
      const Layout& layout);

  template <typename NativeT>
  static Literal CreateFull(absl::Span<const int64_t> dimensions,
                            NativeT value);

  // Returns a tuple literal composed of given literals. Data is copied from the
  // given elements into the returned literal.
  static Literal MakeTuple(absl::Span<const Literal* const> elements);

  static Literal MakeTupleFromSlices(absl::Span<const LiteralSlice> elements);

  // As above, but intended to be invoked with move semantics; i.e.
  //
  //  std::vector<Literal> elements = ...;
  //  auto result = LiteralUtil::MakeTupleOwned(std::move(elements));
  //
  // This would have been declared as an overload, but there is ambiguity
  // in invocation between the above signature and this one.
  static Literal MakeTupleOwned(std::vector<Literal> elements);

  // This overload lets you pass a list of Literals to MakeTupleOwned:
  //
  //   LiteralUtil::MakeTupleOwned(LiteralUtil::CreateR1(...), ...).
  //
  // Simply relying on the MakeTupleOwned(std::vector<Literal>)
  // overload doesn't work because std::initializer_list's elements are always
  // const.
  //
  // The arguments to this function must all be Literal.
  template <typename... Ts>
  static Literal MakeTupleOwned(Ts... elements) {
    std::array<Literal, sizeof...(Ts)> arr{std::move(elements)...};
    std::vector<Literal> v;
    v.insert(v.begin(), std::make_move_iterator(arr.begin()),
             std::make_move_iterator(arr.end()));
    return MakeTupleOwned(std::move(v));
  }
};

// static
template <typename NativeT>
Literal LiteralUtil::CreateR0(NativeT value) {
  Literal literal(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {}));
  literal.Set({}, value);
  return literal;
}

// static
template <typename T>
Literal LiteralUtil::CreateR0(PrimitiveType primitive_type, T value) {
  return primitive_util::ArrayTypeSwitch<Literal>(
      [&value](auto type) {
        using NativeT = primitive_util::NativeTypeOf<type>;
        return CreateR0(static_cast<NativeT>(value));
      },
      primitive_type);
}

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
Literal LiteralUtil::CreateR2WithLayout(
    std::initializer_list<std::initializer_list<NativeT>> values,
    const Layout& layout) {
  Literal literal(ShapeUtil::MakeShapeWithDenseLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {static_cast<int64_t>(values.size()),
       static_cast<int64_t>(values.begin()->size())},
      layout.minor_to_major()));
  literal.PopulateR2(values);
  return literal;
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  return CreateR2WithLayout(values, LayoutUtil::GetDefaultLayoutForR2());
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR3WithLayout(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values,
    const Layout& layout) {
  const int64_t d0 = values.size();
  const int64_t d1 = values.begin()->size();
  const int64_t d2 = values.begin()->begin()->size();
  Array3D<NativeT> tmp(d0, d1, d2);
  int64_t i0 = 0;
  for (auto d1_values : values) {
    int64_t i1 = 0;
    for (auto d2_values : d1_values) {
      int64_t i2 = 0;
      for (auto value : d2_values) {
        tmp(i0, i1, i2) = value;
        ++i2;
      }
      ++i1;
    }
    ++i0;
  }
  return CreateR3FromArray3DWithLayout(tmp, layout);
}

// static
template <typename NativeT>
Literal LiteralUtil::CreateR3(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values) {
  return CreateR3WithLayout(values, LayoutUtil::GetDefaultLayoutForR3());
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
