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

#include "zkx/shape_util.h"

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

#include "xla/tsl/platform/status.h"
#include "zkx/base/logging.h"
#include "zkx/layout_util.h"
#include "zkx/printer.h"

namespace zkx {
namespace {

constexpr int64_t kAnnotationPrintInterval = 5;

inline absl::Status ShapeError(const Shape& shape, std::string_view message) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "Shape Error: %s Shape(%s): %s", message,
      PrimitiveType_IsValid(shape.element_type())
          ? primitive_util::LowercasePrimitiveTypeName(shape.element_type())
          : absl::StrCat(static_cast<int>(shape.element_type())),
      shape.DebugString()));
}

template <bool kPrintLayout>
void PrintShape(Printer* printer, const Shape& shape) {
  if constexpr (kPrintLayout) {
    ShapeUtil::PrintHumanStringWithLayout(printer, shape);
  } else {
    ShapeUtil::PrintHumanString(printer, shape);
  }
}

template <bool kPrintLayout>
void PrintTupleShapes(Printer* printer, absl::Span<const Shape> tuple_shapes) {
  if (ABSL_PREDICT_FALSE(tuple_shapes.empty())) {
    printer->Append("()");
    return;
  }
  printer->Append("(");
  PrintShape<kPrintLayout>(printer, tuple_shapes[0]);
  for (int64_t i = 1; i < tuple_shapes.size(); ++i) {
    if (i % kAnnotationPrintInterval == 0) {
      printer->Append(absl::StrFormat(", /*index=%lld*/", i));
    } else {
      printer->Append(", ");
    }
    PrintShape<kPrintLayout>(printer, tuple_shapes[i]);
  }
  printer->Append(")");
}

// Validates the shape size is sane. This makes sure it's safe to do
// calculations in int64_t without overflowing.
absl::Status ValidateShapeSize(const Shape& shape) {
  if (!shape.IsArray()) {
    return absl::OkStatus();
  }

  auto [extent_product, extent_overflow] =
      ShapeUtil::ExtentProduct</*kBoundedDynamicOk=*/true>(shape);
  auto [dense_shape_size, byte_width_overflow] = OverflowSafeMultiply(
      extent_product, primitive_util::ByteWidth(shape.element_type()));

  if (extent_overflow || byte_width_overflow) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Shape %s size may overflow int64_t.", ShapeUtil::HumanString(shape)));
  }
  return absl::OkStatus();
}

absl::Status ValidateDimensions(const Shape& shape) {
  bool any_overflows = false;
  int64_t product = 1;
  for (int64_t i = 0; i < shape.rank(); ++i) {
    int64_t dimension = shape.dimensions(i);
    if (dimension == Shape::kUnboundedSize) {
      continue;
    }
    if (dimension < 0) {
      return ShapeError(
          shape,
          absl::StrFormat("Negative dimension at index %d: %d.", i, dimension));
    }
    bool overflow;
    std::tie(product, overflow) = OverflowSafeMultiply(product, dimension);
    any_overflows |= overflow;
  }
  if (any_overflows) {
    return ShapeError(shape, "Dimensions overflow.");
  }
  return absl::OkStatus();
}

// Validates all of the non-layout properties of the shape -- this is a helper
// used by both the layout-optional and layout-required public method.
absl::Status ValidateNonLayoutProperties(const Shape& shape) {
  if (shape.element_type() == PRIMITIVE_TYPE_INVALID ||
      !PrimitiveType_IsValid(shape.element_type())) {
    return ShapeError(shape, "Invalid element type.");
  }
  if (shape.element_type() == TUPLE) {
    if (shape.dimensions_size() != 0) {
      return ShapeError(shape, "This type cannot have dimensions.");
    }
    for (auto& element_shape : shape.tuple_shapes()) {
      TF_RETURN_IF_ERROR(ValidateNonLayoutProperties(element_shape));
    }
    return absl::OkStatus();
  }

  // Non-tuple shape.
  if (shape.tuple_shapes_size() > 0) {
    return ShapeError(shape, "Non-tuple type contains tuple_shapes.");
  }

  // Tokens and opaques should not have layout or dimensions.
  if (shape.element_type() == TOKEN || shape.element_type() == OPAQUE_TYPE) {
    if (shape.dimensions_size() != 0) {
      return ShapeError(shape, "This type cannot have dimensions.");
    }
    if (shape.has_layout()) {
      return ShapeError(shape, "This type cannot have a layout.");
    }
    return absl::OkStatus();
  }

  TF_RETURN_IF_ERROR(ValidateDimensions(shape));
  TF_RETURN_IF_ERROR(ValidateShapeSize(shape));
  return absl::OkStatus();
}

template <typename T>
const T& Deref(const T* ptr) {
  DCHECK(ptr != nullptr);
  return *ptr;
}

template <typename T>
const T& Deref(const T& ref) {
  return ref;
}

template <typename ShapePtrOrRef>
Shape MakeTupleShapeImpl(absl::Span<ShapePtrOrRef> shapes) {
  Shape result;
  result.set_element_type(TUPLE);
  result.mutable_tuple_shapes()->reserve(shapes.size());
  for (const auto& shape : shapes) {
    ShapeUtil::AppendShapeToTuple(Deref(shape), &result);
  }
  TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(result));
  return result;
}

}  // namespace

std::string ShapeIndex::ToString() const {
  return absl::StrCat("{", absl::StrJoin(*this, ","), "}");
}

std::ostream& operator<<(std::ostream& out, const ShapeIndex& shape_index) {
  out << shape_index.ToString();
  return out;
}

// static
bool ShapeUtil::Equal(const Shape& lhs, const Shape& rhs) {
  bool equal = Shape::Equal()(lhs, rhs);

  // TODO(chokobole): Uncomment this. Dependency: VLOG_IS_ON
  // if (!equal && VLOG_IS_ON(3)) {
  if (!equal) {
    VLOG(3) << "ShapeUtil::Equal differ: lhs = " << lhs.ShortDebugString()
            << ", rhs = " << rhs.ShortDebugString();
  }

  return equal;
}

// static
bool ShapeUtil::EqualStructure(const Shape& lhs, const Shape& rhs) {
  bool equal = true;
  ForEachSubshape(lhs, [&](const Shape& /*subshape*/, const ShapeIndex& index) {
    equal = equal && IndexIsValid(rhs, index);
  });
  ForEachSubshape(rhs, [&](const Shape& /*subshape*/, const ShapeIndex& index) {
    equal = equal && IndexIsValid(lhs, index);
  });

  return equal;
}

// static
bool ShapeUtil::FillNewShape(PrimitiveType element_type,
                             absl::Span<const int64_t> dimensions,
                             Shape* shape) {
  int64_t dense_shape_size = primitive_util::IsArrayType(element_type)
                                 ? primitive_util::ByteWidth(element_type)
                                 : -1;

  // Verify that array-based lookup is consistent with public API.
  DCHECK_EQ(dense_shape_size, primitive_util::ByteWidth(element_type))
      << element_type;

  shape->set_element_type(element_type);
  const int ndims = dimensions.size();
  auto layout = shape->mutable_layout();
  auto* minor_to_major = layout->mutable_minor_to_major();
  int64_t static_extent_product = dense_shape_size;
  bool any_overflows = false;
  for (int i = 0; i < ndims; i++) {
    const int64_t d = dimensions[i];
    if (d != Shape::kUnboundedSize) {
      bool overflow;
      std::tie(static_extent_product, overflow) =
          OverflowSafeMultiply(static_extent_product, d);
      any_overflows |= overflow;
    }

    shape->add_dimensions(d);
    minor_to_major->push_back(ndims - 1 - i);
  }
  if (any_overflows) {
    return false;
  }
  return true;
}

// static
Shape ShapeUtil::MakeShape(PrimitiveType element_type,
                           absl::Span<const int64_t> dimensions) {
  Shape shape;
  CHECK(FillNewShape(element_type, dimensions, &shape));
  return shape;
}

// static
Shape ShapeUtil::MakeScalarShape(PrimitiveType element_type) {
  return MakeShape(element_type, {});
}

// static
Shape ShapeUtil::MakeTupleShape(absl::Span<const Shape> shapes) {
  return MakeTupleShapeImpl(shapes);
}

// static
Shape ShapeUtil::MakeTupleShapeWithPtrs(absl::Span<const Shape* const> shapes) {
  return MakeTupleShapeImpl(shapes);
}

// static
Shape ShapeUtil::MakeOpaqueShape() {
  Shape result;
  result.set_element_type(OPAQUE_TYPE);
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(result));
  return result;
}

// static
Shape ShapeUtil::MakeTokenShape() {
  Shape result;
  result.set_element_type(TOKEN);
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(result));
  return result;
}

// static
void ShapeUtil::AppendShapeToTuple(const Shape& shape, Shape* tuple_shape) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape));
  *tuple_shape->add_tuple_shapes() = shape;
}

// static
int64_t ShapeUtil::TupleElementCount(const Shape& shape) {
  return shape.tuple_shapes_size();
}

// static
int64_t ShapeUtil::SubshapeCount(const Shape& shape) {
  int64_t n = 0;
  ForEachSubshape(shape, [&](const Shape& literal_subshape,
                             const ShapeIndex& index) { ++n; });
  return n;
}

// static
bool ShapeUtil::IsZeroElementArray(const Shape& shape) {
  return shape.IsArray() && absl::c_linear_search(shape.dimensions(), 0);
}

// static
void ShapeUtil::PrintHumanString(Printer* printer, const Shape& shape) {
  if (shape.IsTuple()) {
    PrintTupleShapes</*kPrintLayout=*/false>(printer, shape.tuple_shapes());
    return;
  }
  printer->Append(
      primitive_util::LowercasePrimitiveTypeName(shape.element_type()));
  if (shape.dimensions().empty()) {
    printer->Append("[]");
    return;
  }
  printer->Append("[");
  auto print_one = [&](int i) {
    if (shape.is_dynamic_dimension(i)) {
      if (shape.dimensions(i) != Shape::kUnboundedSize) {
        printer->Append(absl::StrCat("<=", shape.dimensions(i)));
      } else {
        printer->Append("?");
      }
    } else {
      printer->Append(shape.dimensions(i));
    }
  };
  print_one(0);
  for (int i = 1, n = shape.dimensions_size(); i < n; ++i) {
    printer->Append(",");
    print_one(i);
  }
  printer->Append("]");
}

// static
void ShapeUtil::PrintHumanStringWithLayout(Printer* printer,
                                           const Shape& shape) {
  if (shape.IsTuple()) {
    PrintTupleShapes</*kPrintLayout=*/true>(printer, shape.tuple_shapes());
    return;
  }
  PrintHumanString(printer, shape);
  if (!shape.has_layout()) return;
  if (IsScalar(shape)) {
    std::string layout_str = LayoutUtil::HumanString(shape.layout());
    // Don't print "{}" as layout for scalars.
    if (layout_str != "{}") {
      printer->Append(layout_str);
    }
  } else if (shape.IsArray()) {
    LayoutUtil::PrintHumanString(printer, shape.layout());
  }
}

// static
void ShapeUtil::PrintHumanString(Printer* printer,
                                 const ProgramShape& program_shape) {
  printer->Append("(");
  const auto& shape_parameters = program_shape.parameters();
  if (!shape_parameters.empty()) {
    auto print_one = [&](int i) {
      if (i < program_shape.parameter_names_size()) {
        printer->Append(program_shape.parameter_names(i));
      } else {
        printer->Append("(unknown)");
      }
      printer->Append(": ");
      PrintHumanString(printer, shape_parameters[i]);
    };
    print_one(0);
    for (int i = 1; i < shape_parameters.size(); ++i) {
      printer->Append(", ");
      print_one(i);
    }
  }
  printer->Append(") -> ");
  PrintHumanString(printer, program_shape.result());
}

// static
std::string ShapeUtil::HumanString(const Shape& shape) {
  StringPrinter printer;
  PrintHumanString(&printer, shape);
  return std::move(printer).ToString();
}

// static
std::string ShapeUtil::HumanStringWithLayout(const Shape& shape) {
  StringPrinter printer;
  PrintHumanStringWithLayout(&printer, shape);
  return std::move(printer).ToString();
}

// static
std::string ShapeUtil::HumanString(const ProgramShape& program_shape) {
  StringPrinter printer;
  PrintHumanString(&printer, program_shape);
  return std::move(printer).ToString();
}

// static
bool ShapeUtil::Compatible(const Shape& lhs, const Shape& rhs) {
  return Shape::Equal().IgnoreDynamicDimension().IgnoreLayout()(lhs, rhs);
}

// static
bool ShapeUtil::CompatibleKind(const Shape& lhs, const Shape& rhs) {
  return Shape::Equal()
      .IgnoreElementType()
      .IgnoreLayout()
      .IgnoreDimensions()
      .IgnoreDynamicDimension()(lhs, rhs);
}

// static
int64_t ShapeUtil::ByteSizeOfPrimitiveType(PrimitiveType primitive_type) {
  return primitive_util::ByteWidth(primitive_type);
}

// static
int64_t ShapeUtil::ByteSizeOf(const Shape& shape, int64_t pointer_size) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape));
  if (shape.element_type() == TUPLE) {
    return ByteSizeOfTupleIndexTable(shape, pointer_size);
  } else if (shape.IsArray()) {
    return ByteSizeOfElements(shape);
  } else if (shape.element_type() == TOKEN) {
    return 0;
  } else if (shape.element_type() == OPAQUE_TYPE) {
    CHECK_GT(pointer_size, 0);
    return pointer_size;
  }
  LOG(FATAL) << PrimitiveType_Name(shape.element_type())
             << " primitive type has no definitive size";
}

// static
int64_t ShapeUtil::ByteSizeOfTupleIndexTable(const Shape& shape,
                                             int64_t pointer_size) {
  TF_DCHECK_OK(ValidateShape(shape));
  CHECK_EQ(TUPLE, shape.element_type());
  CHECK_GT(pointer_size, 0);
  return pointer_size * shape.tuple_shapes_size();
}

int64_t ShapeUtil::ByteSizeOfElements(const Shape& shape) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape));
  int64_t allocated_element_count;

  CHECK(LayoutUtil::IsDenseArray(shape)) << shape.ShortDebugString();
  allocated_element_count = ElementsIn(shape);

  if (shape.has_layout() && shape.layout().element_size_in_bits() != 0) {
    const int64_t num_bits =
        allocated_element_count * shape.layout().element_size_in_bits();
    return tsl::MathUtil::CeilOfRatio<int64_t>(num_bits, CHAR_BIT);
  }
  return allocated_element_count *
         ByteSizeOfPrimitiveType(shape.element_type());
}

// static
absl::Status ShapeUtil::ValidateShapeWithOptionalLayout(const Shape& shape) {
  TF_RETURN_IF_ERROR(ValidateNonLayoutProperties(shape));

  return LayoutUtil::ValidateLayoutInShape(shape,
                                           /*allow_missing_layouts=*/true);
}

// static
absl::Status ShapeUtil::ValidateShape(const Shape& shape) {
  TF_RETURN_IF_ERROR(ValidateNonLayoutProperties(shape));

  return LayoutUtil::ValidateLayoutInShape(shape);
}

// static
bool ShapeUtil::IndexIsValid(const Shape& shape, ShapeIndexView index) {
  const Shape* subshape = &shape;
  for (auto i : index) {
    if (!subshape->IsTuple() || i >= subshape->tuple_shapes_size() || i < 0) {
      return false;
    }
    subshape = &subshape->tuple_shapes(i);
  }
  return true;
}

// static
const Shape& ShapeUtil::GetSubshape(const Shape& shape, ShapeIndexView index) {
  const Shape* return_shape = &shape;
  for (auto i : index) {
    CHECK(return_shape->IsTuple())
        << "Invalid index " << ShapeIndex(index) << " for shape " << shape;
    return_shape = &return_shape->tuple_shapes(i);
  }
  return *return_shape;
}

// static
const Shape& ShapeUtil::GetSubshapeOneIndex(const Shape& shape, int64_t index) {
  const Shape* return_shape = &shape;
  CHECK(return_shape->IsTuple())
      << "Invalid index " << index << " for shape " << shape;
  return_shape = &return_shape->tuple_shapes(index);
  return *return_shape;
}

// static
absl::StatusOr<const Shape*> ShapeUtil::TryGetSubshape(const Shape& shape,
                                                       ShapeIndexView index) {
  const Shape* return_shape = &shape;
  for (auto i : index) {
    if (!return_shape->IsTuple() || i < 0 ||
        i >= return_shape->tuple_shapes_size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Shape index %s not a valid subshape index for tuple with shape %s",
          ShapeIndex(index).ToString(), shape.DebugString()));
    }
    return_shape = &return_shape->tuple_shapes(i);
  }
  return return_shape;
}

// static
Shape* ShapeUtil::GetMutableSubshape(Shape* shape, ShapeIndexView index) {
  Shape* return_shape = shape;
  for (auto i : index) {
    CHECK(return_shape->IsTuple());
    return_shape = return_shape->mutable_tuple_shapes(i);
  }
  return return_shape;
}

// static
bool ShapeUtil::IsLeafIndex(const Shape& shape, const ShapeIndex& index) {
  return !GetSubshape(shape, index).IsTuple();
}

// static
int64_t ShapeUtil::GetLeafCountTuple(const Shape& shape) {
  DCHECK(shape.IsTuple());
  int64_t count = 0;
  for (const Shape& subshape : shape.tuple_shapes()) {
    if (subshape.IsTuple()) {
      count += GetLeafCount(subshape);
    } else {
      ++count;
    }
  }
  return count;
}

// static
int64_t ShapeUtil::GetLeafCount(const Shape& shape) {
  if (!shape.IsTuple()) {
    return 1;
  }
  return GetLeafCountTuple(shape);
}

// static
bool ShapeUtil::DynamicArrayShapeIsCompatible(const Shape& dynamic_shape,
                                              const Shape& bounded_shape) {
  if (dynamic_shape.rank() != bounded_shape.rank()) {
    return false;
  }
  for (int64_t i = 0; i < dynamic_shape.rank(); ++i) {
    if (dynamic_shape.dimensions(i) > bounded_shape.dimensions(i)) {
      return false;
    }
  }
  return true;
}

// static
bool ShapeUtil::DynamicShapeIsCompatible(const Shape& dynamic_shape,
                                         const Shape& bounded_shape) {
  bool compatible = true;
  ShapeUtil::ForEachSubshape(
      dynamic_shape, [&](const Shape& sub_shape, const ShapeIndex& index) {
        if (compatible) {
          auto subshape_result = TryGetSubshape(bounded_shape, index);
          if (subshape_result.ok()) {
            const Shape* bounded_sub_shape = std::move(subshape_result).value();
            if (sub_shape.IsTuple()) {
              if (!bounded_sub_shape->IsTuple()) {
                compatible = false;
              }
            } else {
              if (bounded_sub_shape->IsTuple()) {
                compatible = false;
              } else if (!sub_shape.is_static() &&
                         !DynamicArrayShapeIsCompatible(sub_shape,
                                                        *bounded_sub_shape)) {
                compatible = false;
              }
            }
          } else {
            compatible = false;
          }
        }
      });
  return compatible;
}

}  // namespace zkx
