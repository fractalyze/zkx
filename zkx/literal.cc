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

#include "zkx/literal.h"

#include "xla/tsl/platform/mem.h"
#include "xla/tsl/platform/status.h"
#include "zkx/primitive_util.h"

namespace zkx {

namespace {

// TODO(chokobole): Update the comment below if we don't have HloEvaluator in
// the end.
// Lazy getter for the interned scalar shape in static storage. We reuse this
// shape pointer to when constructing scalar Literals, which can happen a lot
// when we are evaluating reduce-like ops in HloEvaluator, and copying the
// shape over and over again significantly slows down the evaluator.
template <PrimitiveType kType>
const Shape& ScalarShapeImpl() {
  static_assert(primitive_util::IsArrayType(kType),
                "Not a valid type for a scalar.");
  static const Shape* shape = [] {
    auto shape = new Shape(kType, {}, {}, {});
    shape->mutable_layout();
    return shape;
  }();
  return *shape;
}

const Shape& ScalarShape(PrimitiveType type) {
  return primitive_util::ArrayTypeSwitch<const Shape&>(
      [&](auto primitive_type_constant) -> const Shape& {
        return ScalarShapeImpl<primitive_type_constant>();
      },
      type);
}

const Shape& NilShape() {
  static const Shape* shape = new Shape(TUPLE, {}, {}, {});
  return *shape;
}

// Returns the interned shape pointer in static storage if it's a scalar shape
// or nil shape.
const Shape* TryInternShape(const Shape& shape) {
  if (shape.IsTuple() && shape.tuple_shapes_size() == 0) {
    return &NilShape();
  }
  if (shape.IsArray() && shape.dimensions_size() == 0 && shape.is_static() &&
      shape.has_layout() && shape.layout().tiles_size() == 0 &&
      shape.layout().memory_space() == 0 &&
      shape.layout().element_size_in_bits() == 0) {
    return &ScalarShape(shape.element_type());
  }
  return nullptr;
}

}  // namespace

LiteralBase::~LiteralBase() = default;

Literal LiteralBase::Clone() const {
  Literal result(shape());
  TF_CHECK_OK(result.CopyFrom(*this));
  return result;
}

std::unique_ptr<Literal> LiteralBase::CloneToUnique() const {
  auto result = std::make_unique<Literal>(shape());
  TF_CHECK_OK(result->CopyFrom(*this));
  return result;
}

template <typename NativeT>
bool LiteralBase::Piece::EqualElementsInternal(
    const LiteralBase::Piece& other, std::vector<int64_t>* multi_index) const {
  if (multi_index->size() == subshape().rank()) {
    return (Get<NativeT>(*multi_index) == other.Get<NativeT>(*multi_index));
  }
  for (int64_t i = 0; i < GetDynamicSize(multi_index->size()); ++i) {
    multi_index->push_back(i);
    if (!EqualElementsInternal<NativeT>(other, multi_index)) {
      return false;
    }
    multi_index->pop_back();
  }
  return true;
}

bool LiteralBase::Piece::EqualElements(const LiteralBase::Piece& other) const {
  if (subshape().is_static() &&
      ShapeUtil::Equal(subshape(), other.subshape()) && subshape().IsArray()) {
    CHECK(LayoutUtil::IsDenseArray(subshape()))
        << __func__ << " is only supported for dense arrays: " << subshape();
    CHECK_EQ(size_bytes_dense(), other.size_bytes_dense());
    if (primitive_util::IsSubByteNonPredType(subshape().element_type())) {
      auto one_array = buffer();
      auto two_array = other.buffer();
      const int bits_per_element =
          primitive_util::BitWidth(subshape().element_type());
      const uint8_t mask = LsbMask<uint8_t>(bits_per_element);
      for (int64_t i = 0; i < size_bytes_dense(); ++i) {
        if ((one_array[i] & mask) != (two_array[i] & mask)) return false;
      }
      return true;
    }
    return memcmp(buffer(), other.buffer(), size_bytes_dense()) == 0;
  }

  std::vector<int64_t> multi_index;
  return primitive_util::ArrayTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        using NativeSrcT =
            primitive_util::NativeTypeOf<primitive_type_constant>;
        return EqualElementsInternal<NativeSrcT>(other, &multi_index);
      },
      subshape().element_type());
}

bool LiteralBase::Equal(const LiteralBase& other, bool layout_sensitive) const {
  // Checking the structure of tuple literals. Checks for dense arrays are
  // performed below.
  if (!ShapeUtil::EqualStructure(shape(), other.shape())) {
    return false;
  }

  return root_piece().ForEachSubpieceWithBool([&](const ShapeIndex& index,
                                                  const Piece& piece) {
    const Piece& other_piece = other.piece(index);
    const Shape& subshape = piece.subshape();
    const Shape& other_subshape = other_piece.subshape();
    if (subshape.element_type() != other_subshape.element_type()) {
      return false;
    }
    if (!piece.subshape().IsArray()) {
      return true;
    }
    if (subshape.rank() != other_subshape.rank()) {
      return false;
    }
    if (layout_sensitive && (subshape.layout() != other_subshape.layout())) {
      return false;
    }

    for (int64_t i = 0; i < subshape.rank(); ++i) {
      if (piece.GetDynamicSize(i) != other_piece.GetDynamicSize(i)) {
        return false;
      }
    }

    if (!piece.EqualElements(other_piece)) {
      return false;
    }
    return true;
  });
}

namespace {

template <typename NativeT>
bool AllElementsEqualValue(absl::Span<const NativeT> data, NativeT value) {
  for (int64_t i = 0; i < data.size(); ++i) {
    if (memcmp(&data[i], &value, sizeof value)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool Literal::Piece::IsAll(const Literal& scalar) const {
  CHECK(ShapeUtil::IsScalar(scalar.shape())) << scalar.shape().ToString();
  if (!subshape().IsArray()) {
    return false;
  }

  CHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  CHECK_EQ(subshape().element_type(), scalar.shape().element_type());
  return primitive_util::ArrayTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        return AllElementsEqualValue(this->data<NativeT>(),
                                     scalar.GetFirstElement<NativeT>());
      },
      subshape().element_type());
}

bool LiteralBase::IsAll(const Literal& scalar) const {
  return root_piece().IsAll(scalar);
}

bool LiteralBase::IsAll(int8_t value) const {
  if (!shape().IsArray()) {
    return false;
  }
  PrimitiveType ty = shape().element_type();
  if (primitive_util::IsUnsignedIntegralType(ty) && value < 0) {
    return false;
  }
  Literal scalar(ShapeUtil::MakeScalarShape(ty));
  return primitive_util::ArrayTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        NativeT converted(value);
        if (static_cast<int8_t>(converted) != value) {
          return false;
        }
        scalar.Set<NativeT>({}, converted);
        return root_piece().IsAll(scalar);
      },
      ty);
}

const Shape& LiteralBase::shape() const { return root_piece().subshape(); }

const char* LiteralBase::Piece::buffer() const {
  // std::visit is avoided here due to its code size issues.
  if (auto* r = std::get_if<DenseRep>(&rep_)) {
    return r->data;
  }
  if (auto* r = std::get_if<DenseInlinedRep>(&rep_)) {
    return r->data;
  }
  DCHECK(std::holds_alternative<TupleRep>(rep_) ||
         std::holds_alternative<Uninitialized>(rep_));
  return nullptr;
}

const LiteralBase::Piece& LiteralBase::piece(
    const ShapeIndex& shape_index) const {
  const Piece* piece = &root_piece();
  for (const auto i : shape_index) {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, piece->children_size());
    piece = &piece->child(i);
  }
  return *piece;
}

Shape* MutableLiteralBase::mutable_shape_do_not_use() {
  const Shape* const_shape = shape_.get();
  if (!shape_.OwnsPtr()) {
    shape_ = MaybeOwningShapePtr(std::make_unique<Shape>(*shape_));
  }
  Shape* shape = shape_.get_mutable();

  if (shape != const_shape) {
    std::function<void(const Shape&, Piece*)> set_piece_shapes =
        [&set_piece_shapes](const Shape& shape, Piece* piece) {
          piece->set_subshape(&shape);
          if (shape.IsTuple()) {
            for (int i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
              const Shape& subshape = shape.tuple_shapes(i);
              set_piece_shapes(subshape, &piece->child(i));
            }
          }
        };
    set_piece_shapes(*shape, &mutable_root_piece());
  }
  return shape;
}

Literal::Literal() : Literal(NilShape()) {}

Literal::Literal(const Shape& shape)
    : Literal(shape, /*allocate_arrays=*/true) {}

void Literal::SetShape(const Shape& shape) {
  if (const Shape* intered_shape_ptr = TryInternShape(shape)) {
    shape_ = intered_shape_ptr;
    return;
  }
  auto owning_shape_ptr = std::make_unique<Shape>(shape);
  if (owning_shape_ptr->IsArray() && !owning_shape_ptr->has_layout()) {
    *owning_shape_ptr->mutable_layout() =
        LayoutUtil::GetDefaultLayoutForShape(*owning_shape_ptr);
  }
  if (owning_shape_ptr->IsArray() &&
      LayoutUtil::HasCustomElementSizeInBits(*owning_shape_ptr)) {
    owning_shape_ptr->mutable_layout()->set_element_size_in_bits(0);
  }
  shape_ = std::move(owning_shape_ptr);
}

void Literal::SetPiece(const Shape& shape, Piece* piece, bool allocate_arrays,
                       ArrayValueState leaf_array_value_state) {
  if (shape.IsTuple()) {
    for (const Shape& subshape : shape.tuple_shapes()) {
      Piece child_piece;
      child_piece.set_subshape(&subshape);

      SetPiece(subshape, &child_piece, allocate_arrays, leaf_array_value_state);

      piece->emplace_back(std::move(child_piece));
    }
  } else if (shape.IsArray()) {
    DCHECK(LayoutUtil::IsDenseArray(shape))
        << "literal array storage is currently only supported for dense "
           "arrays: "
        << shape;
    piece->set_array_value_state(leaf_array_value_state);
    if (leaf_array_value_state == LiteralBase::ArrayValueState::kKnown &&
        allocate_arrays) {
      piece->AllocateBuffers();
    }
  }
}

Literal::Literal(const Shape& shape, bool allocate_arrays,
                 ArrayValueState leaf_array_value_state) {
  SetShape(shape);
  CHECK(leaf_array_value_state != ArrayValueState::kKnown ||
        LayoutUtil::HasLayout(*shape_));
  root_piece_.set_subshape(shape_.get());
  CHECK(&root_piece_.subshape() == shape_.get());

  SetPiece(*shape_, &root_piece_, allocate_arrays, leaf_array_value_state);
}

Literal::~Literal() { DeallocateBuffers(); }

void Literal::DeallocateBuffers() {
  root_piece_.ForEachMutableSubpiece(
      [&](const ShapeIndex& index, Piece* piece) {
        piece->DeallocateBuffers();
      });
}

Literal::Literal(Literal&& other) { *this = std::move(other); }

Literal& Literal::operator=(Literal&& other) {
  DCHECK(&other.root_piece_.subshape() == other.shape_.get());
  using std::swap;
  swap(shape_, other.shape_);
  swap(root_piece_, other.root_piece_);
  DCHECK(&root_piece_.subshape() == shape_.get());

  return *this;
}

int32_t LiteralBase::GetDynamicSize(int64_t dim_index) const {
  return GetDynamicSize(dim_index, {});
}

int32_t LiteralBase::GetDynamicSize(int64_t dim_index,
                                    const ShapeIndex& shape_index) const {
  return piece(shape_index).GetDynamicSize(dim_index);
}

std::optional<int64_t> LiteralBase::GetFirstInteger() const {
  if (!primitive_util::IsIntegralType(shape().element_type())) {
    return std::nullopt;
  }
  return primitive_util::IntegralTypeSwitch<std::optional<int64_t>>(
      [&](auto primitive_type_constant) -> std::optional<int64_t> {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        auto first_element = GetFirstElement<NativeT>();
        if constexpr (std::is_same_v<NativeT, uint64_t>) {
          int64_t v = static_cast<int64_t>(first_element);
          if (v < 0) {
            return std::nullopt;
          }
        }
        return first_element;
      },
      shape().element_type());
}

namespace {

// Copies the elements in 'src' to 'dest'. The shape and layout of the data in
// the array slices are indicated by dest_shape and src_shape respectively.
template <typename NativeT>
void CopyElementsBetween(absl::Span<NativeT> dest,
                         absl::Span<const NativeT> src, const Shape& dest_shape,
                         const Shape& src_shape) {
  DCHECK(LayoutUtil::IsDenseArray(dest_shape));
  DCHECK(LayoutUtil::IsDenseArray(src_shape));
  DCHECK(ShapeUtil::Compatible(dest_shape, src_shape));
  if (ShapeUtil::IsZeroElementArray(dest_shape)) {
    return;
  }
  std::vector<int64_t> index(dest_shape.rank());
  do {
    dest[IndexUtil::MultidimensionalIndexToLinearIndex(dest_shape, index)] =
        src[IndexUtil::MultidimensionalIndexToLinearIndex(src_shape, index)];
  } while (IndexUtil::BumpIndices(dest_shape, absl::MakeSpan(index)));
}

}  // namespace

int32_t LiteralBase::Piece::GetDynamicSize(int64_t dim_index) const {
  CHECK(LayoutUtil::IsDenseArray(subshape()));
  if (!subshape_->is_dynamic_dimension(dim_index)) {
    // This is a static dimension, return size.
    return subshape_->dimensions(dim_index);
  }
  return dynamic_size_buffer()[dim_index];
}

void LiteralBase::Piece::SetDynamicSize(int64_t dim_index, int32_t size) {
  CHECK(LayoutUtil::IsDenseArray(subshape()));
  CHECK(subshape_->is_dynamic_dimension(dim_index));
  dynamic_size_buffer()[dim_index] = size;
}

void LiteralBase::Piece::AllocateBuffers() {
  const int64_t bytes = total_bytes_dense();
  if (bytes > kMaxInlinedBytes) {
    CHECK_EQ(buffer(), nullptr);
    rep_.emplace<DenseRep>();
    char* buffer =
        static_cast<char*>(tsl::port::AlignedMalloc(bytes, kMinimumAlignment));
    CHECK(buffer != nullptr) << "Failed to allocate buffer for Literal";
    set_buffer(buffer);
  } else {
    rep_.emplace<DenseInlinedRep>();
  }
}

void LiteralBase::Piece::DeallocateBuffers() {
  if (auto* array_rep = GetDenseRep()) {
    tsl::port::AlignedFree(array_rep->data);
    rep_.emplace<Uninitialized>();
  }
}

template <typename NativeT>
void LiteralBase::Piece::CopyElementsWithDynamicBound(
    const LiteralBase::Piece& src) {
  auto& dest_shape = subshape();
  auto& src_shape = src.subshape();

  // At least one shape has to be static as bound.
  CHECK(dest_shape.is_static() || src_shape.is_static());
  auto& bound_shape = dest_shape.is_static() ? src_shape : dest_shape;
  if (ShapeUtil::IsZeroElementArray(dest_shape)) {
    return;
  }
  if (dest_shape.rank() == 1) {
    // Fast path for rank 1 arrays.
    int64_t count = std::min(GetDynamicSize(0), src.GetDynamicSize(0));
    std::copy_n(src.data<NativeT>().begin(), count, data<NativeT>().begin());
    return;
  }
  std::vector<int64_t> index(dest_shape.rank());
  do {
    bool out_of_bound = false;
    for (int64_t i = 0; i < index.size(); ++i) {
      // Do not copy elements beyond dynamic bound.
      if (index[i] >= GetDynamicSize(i) || index[i] >= src.GetDynamicSize(i)) {
        out_of_bound = true;
      }
    }
    if (out_of_bound) {
      continue;
    }
    data<NativeT>()[IndexUtil::MultidimensionalIndexToLinearIndex(dest_shape,
                                                                  index)] =
        src.data<NativeT>()[IndexUtil::MultidimensionalIndexToLinearIndex(
            src_shape, index)];
  } while (IndexUtil::BumpIndices(bound_shape, absl::MakeSpan(index)));
}

absl::Status LiteralBase::Piece::CopyFrom(const LiteralBase::Piece& src,
                                          bool only_dynamic_bound) {
  CHECK(subshape_ != nullptr);
  CHECK(src.subshape_ != nullptr);
  CHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  CHECK(LayoutUtil::IsDenseArray(src.subshape()))
      << __func__ << " is only supported for dense arrays: " << src.subshape();
  if (!only_dynamic_bound) {
    CHECK(ShapeUtil::Compatible(subshape(), src.subshape()));
  }
  if (src.array_value_state_ == ArrayValueState::kUnknown ||
      src.array_value_state_ == ArrayValueState::kUndetermined) {
    if (array_value_state_ == ArrayValueState::kKnown) {
      DeallocateBuffers();
    }
    array_value_state_ = src.array_value_state_;
    return absl::OkStatus();
  } else {
    CHECK(src.array_value_state_ == ArrayValueState::kKnown);
    if (array_value_state_ == ArrayValueState::kUndetermined ||
        array_value_state_ == ArrayValueState::kUnknown) {
      AllocateBuffers();
    }
    array_value_state_ = src.array_value_state_;
  }

  if (ShapeUtil::Equal(subshape(), src.subshape())) {
    // If the layouts are equal it's faster just to memcpy.
    memcpy(buffer(), src.buffer(), src.size_bytes_dense());
  } else {
    std::vector<int64_t> origin(subshape().rank(), 0);
    primitive_util::ArrayTypeSwitch<void>(
        [&](auto primitive_type_constant) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          if (only_dynamic_bound) {
            CopyElementsWithDynamicBound<NativeT>(src);
          } else {
            CopyElementsBetween<NativeT>(this->data<NativeT>(),
                                         src.data<NativeT>(), subshape(),
                                         src.subshape());
          }
        },
        subshape().element_type());
  }
  DCHECK_EQ(dynamic_size_buffer_bytes(), src.dynamic_size_buffer_bytes());
  if (subshape().is_dynamic() && src.subshape().is_dynamic()) {
    memcpy(dynamic_size_buffer(), src.dynamic_size_buffer(),
           src.dynamic_size_buffer_bytes());
  }
  return absl::OkStatus();
}

const void* LiteralBase::Piece::untyped_data() const {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << ShapeUtil::HumanString(subshape());
  return buffer();
}

void* LiteralBase::Piece::untyped_data() {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << ShapeUtil::HumanString(subshape());
  return buffer();
}

const void* LiteralBase::untyped_data(const ShapeIndex& shape_index) const {
  return piece(shape_index).untyped_data();
}

void* MutableLiteralBase::untyped_data(const ShapeIndex& shape_index) {
  return piece(shape_index).untyped_data();
}

int64_t LiteralBase::size_bytes(const ShapeIndex& shape_index) const {
  return piece(shape_index).size_bytes_dense();
}

MutableLiteralBase::~MutableLiteralBase() = default;

void MutableLiteralBase::SetDynamicSize(int64_t dim_index, int32_t size) {
  return SetDynamicSize(dim_index, {}, size);
}

void MutableLiteralBase::SetDynamicSize(int64_t dim_index,
                                        const ShapeIndex& shape_index,
                                        int32_t size) {
  Shape* subshape =
      ShapeUtil::GetMutableSubshape(mutable_shape_do_not_use(), shape_index);
  CHECK(LayoutUtil::IsDenseArray(*subshape))
      << __func__ << " is only supported for dense arrays: " << *subshape;
  CHECK_GE(subshape->dimensions(dim_index), size);
  subshape->set_dynamic_dimension(dim_index, true);
  CHECK_EQ(&piece(shape_index).subshape(), subshape);

  piece(shape_index).SetDynamicSize(dim_index, size);
}

absl::Status Literal::MoveFrom(Literal&& src_literal,
                               const ShapeIndex& dest_shape_index) {
  const Shape& dest_subshape =
      ShapeUtil::GetSubshape(shape(), dest_shape_index);
  if (!ShapeUtil::Equal(dest_subshape, src_literal.shape())) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Destination subshape not equal to source shape: %s vs %s",
        ShapeUtil::HumanString(dest_subshape),
        ShapeUtil::HumanString(src_literal.shape())));
  }

  src_literal.root_piece_.ForEachMutableSubpiece(
      [&](const ShapeIndex& src_index, Piece* src_piece) {
        if (!src_piece->subshape().IsArray()) {
          return;
        }

        ShapeIndex dest_index = dest_shape_index;
        for (int64_t i : src_index) {
          dest_index.push_back(i);
        }
        Piece& dest_piece = piece(dest_index);
        dest_piece.DeallocateBuffers();
        dest_piece.MoveDataFrom(*src_piece);
      });

  src_literal.shape_ = MaybeOwningShapePtr(&NilShape());
  src_literal.root_piece_ = Piece();
  src_literal.root_piece_.set_subshape(src_literal.shape_.get());

  return absl::OkStatus();
}

absl::Status MutableLiteralBase::CopyFrom(const LiteralSlice& src_literal,
                                          const ShapeIndex& dest_shape_index,
                                          const ShapeIndex& src_shape_index,
                                          bool only_dynamic_bound) {
  const Shape& dest_subshape =
      ShapeUtil::GetSubshape(shape(), dest_shape_index);
  const Shape& src_subshape =
      ShapeUtil::GetSubshape(src_literal.shape(), src_shape_index);
  if (only_dynamic_bound) {
    auto& bound_shape =
        dest_subshape.is_static() ? src_subshape : dest_subshape;
    auto& compact_shape =
        dest_subshape.is_static() ? dest_subshape : src_subshape;
    CHECK(ShapeUtil::DynamicShapeIsCompatible(compact_shape, bound_shape))
        << compact_shape.ToString() << " vs " << bound_shape.ToString();
  } else {
    if (!ShapeUtil::Compatible(dest_subshape, src_subshape)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Destination subshape incompatible with source subshape: %s vs %s",
          ShapeUtil::HumanString(dest_subshape),
          ShapeUtil::HumanString(src_subshape)));
    }
  }
  return mutable_root_piece().ForEachMutableSubpieceWithStatus(
      [&](const ShapeIndex& index, Piece* piece) {
        if (!piece->subshape().IsArray()) {
          return absl::OkStatus();
        }

        // Determine if this index is in the part of this literal that we want
        // to copy over from src_literal.
        bool in_subtree_to_copy = true;
        for (int i = 0; i < dest_shape_index.size(); ++i) {
          if (index[i] != dest_shape_index[i]) {
            in_subtree_to_copy = false;
            break;
          }
        }
        if (!in_subtree_to_copy) {
          return absl::OkStatus();
        }
        // Construct the index of the corresponding piece in the source literal.
        ShapeIndex src_piece_index = src_shape_index;
        for (int64_t i = dest_shape_index.size(), end = index.size(); i < end;
             ++i) {
          src_piece_index.push_back(index[i]);
        }
        TF_RETURN_IF_ERROR(
            piece->CopyFrom(src_literal.piece(src_piece_index),
                            /*only_dynamic_bound=*/only_dynamic_bound));
        return absl::OkStatus();
      });
}

}  // namespace zkx
