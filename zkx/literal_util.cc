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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "xla/tsl/platform/status.h"

namespace zkx {
namespace {

template <PrimitiveType kType>
using NativeT = typename primitive_util::PrimitiveTypeToNative<kType>::type;

template <PrimitiveType kType, typename F, typename... Args>
Literal CreateScalarImpl(F&& value_provider, Args... args) {
  return LiteralUtil::CreateR0<NativeT<kType>>(
      value_provider(std::forward<Args>(args)...));
}

template <template <PrimitiveType> class F, typename... Args>
Literal CreateScalar(PrimitiveType primitive_type, Args... args) {
  return primitive_util::PrimitiveTypeSwitch<Literal>(
      [&](auto primitive_type_constant) -> Literal {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          return CreateScalarImpl<primitive_type_constant>(
              F<primitive_type_constant>{}, std::forward<Args>(args)...);
        }
        LOG(FATAL) << "Unhandled primitive type " << primitive_type;
      },
      primitive_type);
}

template <PrimitiveType kType>
struct ZeroProvider {
  NativeT<kType> operator()() const { return static_cast<NativeT<kType>>(0); }
};

template <PrimitiveType kType>
struct OneProvider {
  NativeT<kType> operator()() const { return static_cast<NativeT<kType>>(1); }
};

}  // namespace

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

// static
std::string LiteralUtil::MultiIndexAsString(
    absl::Span<const int64_t> multi_index) {
  return absl::StrCat("{", absl::StrJoin(multi_index, ","), "}");
}

absl::StatusOr<Literal> MakeFakeLiteral(const Shape& shape,
                                        bool pseudo_random) {
  auto engine = pseudo_random ? std::make_unique<std::minstd_rand0>() : nullptr;
  return MakeFakeLiteral(shape, engine.get(), /*limit=*/std::nullopt,
                         /*is_sorted=*/false,
                         /*no_duplicates=*/false,
                         /*max_bits_of_precision=*/std::nullopt);
}

namespace {

// uniform_int_distribution is not defined for 8-bit integers.
// Use 'short' for those types.
template <typename IntT>
using RngT = std::conditional_t<
    sizeof(IntT) < sizeof(uint16_t),
    std::conditional_t<std::numeric_limits<IntT>::is_signed, int16_t, uint16_t>,
    IntT>;
template <typename IntT>
void PopulateWithRandomIntegralDataWithBounds(Literal* literal,
                                              std::minstd_rand0* engine,
                                              bool no_duplicates, IntT min,
                                              IntT max) {
  CHECK(engine != nullptr);
  CHECK_EQ(literal->shape().element_type(),
           primitive_util::NativeToPrimitiveType<IntT>());
  if (no_duplicates &&
      ShapeUtil::ElementsIn(literal->shape()) < static_cast<int64_t>(max)) {
    std::iota(literal->data<IntT>().begin(), literal->data<IntT>().end(),
              static_cast<IntT>(0));
    std::shuffle(literal->data<IntT>().begin(), literal->data<IntT>().end(),
                 *engine);
  } else {
    std::uniform_int_distribution<RngT<IntT>> generator(
        static_cast<RngT<IntT>>(min), static_cast<RngT<IntT>>(max));
    for (IntT& value : literal->data<IntT>()) {
      value = static_cast<IntT>(generator(*engine));
    }
  }
}

}  // namespace

// static
Literal LiteralUtil::Zero(PrimitiveType primitive_type) {
  return CreateScalar<ZeroProvider>(primitive_type);
}

// static
Literal LiteralUtil::One(PrimitiveType primitive_type) {
  return CreateScalar<OneProvider>(primitive_type);
}

absl::StatusOr<Literal> MakeFakeLiteral(
    const Shape& shape, std::minstd_rand0* engine,
    std::optional<std::pair<int64_t, int64_t>> limit, bool is_sorted,
    bool no_duplicates, std::optional<int64_t> max_bits_of_precision) {
  if (shape.IsTuple()) {
    std::vector<Literal> elements;
    const auto& shape_tuple_shapes = shape.tuple_shapes();
    elements.reserve(shape_tuple_shapes.size());
    for (const Shape& element_shape : shape_tuple_shapes) {
      TF_ASSIGN_OR_RETURN(
          Literal element,
          MakeFakeLiteral(element_shape, engine, limit, is_sorted,
                          no_duplicates, max_bits_of_precision));
      elements.push_back(std::move(element));
    }
    return LiteralUtil::MakeTupleOwned(std::move(elements));
  }
  if (engine == nullptr) {
    return Literal::CreateFromShape(shape);
  }
  // Clear tiles/element size in shape's layout before using it for creating
  // literal.
  Shape new_shape = shape;
  new_shape.mutable_layout()->clear_tiles();
  new_shape.mutable_layout()->set_tail_padding_alignment_in_elements(1);
  new_shape.mutable_layout()->set_element_size_in_bits(0);
  Literal literal(new_shape);

  TF_RETURN_IF_ERROR(primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          if constexpr (primitive_type_constant == PRED) {
            std::uniform_int_distribution<int> generator(0, 1);
            TF_CHECK_OK(literal.Populate<bool>(
                [&](absl::Span<const int64_t> /*indices*/) {
                  return generator(*engine);
                }));
            return absl::OkStatus();
          }
          if constexpr (primitive_util::IsIntegralType(
                            primitive_type_constant)) {
            NativeT max = std::numeric_limits<NativeT>::max();
            NativeT min = std::numeric_limits<NativeT>::lowest();
            if (limit.has_value()) {
              max = static_cast<NativeT>(limit->second);
              min = static_cast<NativeT>(limit->first);
            }
            if (max_bits_of_precision.has_value()) {
              max = std::min(max,
                             static_cast<NativeT>(1 << *max_bits_of_precision));
              if (primitive_util::IsSignedIntegralType(
                      primitive_type_constant)) {
                min = std::max(
                    min, static_cast<NativeT>(-(1 << *max_bits_of_precision)));
              }
            }
            PopulateWithRandomIntegralDataWithBounds<NativeT>(
                &literal, engine, no_duplicates, min, max);
            if (is_sorted) {
              std::sort(literal.data<NativeT>().begin(),
                        literal.data<NativeT>().end());
            }
            return absl::OkStatus();
          }
        }
        return absl::UnimplementedError(absl::StrFormat(
            "Unsupported type for fake random literal generation with bounds: "
            "%s",
            ShapeUtil::HumanString(shape)));
      },
      shape.element_type()));
  return std::move(literal);
}

}  // namespace zkx
