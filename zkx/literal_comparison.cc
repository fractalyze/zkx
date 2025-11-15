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

#include "zkx/literal_comparison.h"

#include <stdint.h>

#include "absl/log/vlog_is_on.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/base/logging.h"
#include "zkx/literal_util.h"
#include "zkx/primitive_util.h"
#include "zkx/shape_util.h"
#include "zkx/util.h"

namespace zkx::literal_comparison {
namespace {

// Templated comparator that specializes for float equality comparison with the
// bitwise helper above (this is the un-specialized fallback, to just use the
// default gunit implementation).
template <typename NativeT>
bool CompareEqual(NativeT lhs, NativeT rhs,
                  absl::Span<const int64_t> multi_index) {
  // Specializations for floating types that do bitwise comparisons when
  // equality comparison is requested.
  return lhs == rhs;
}

template <typename NativeT>
absl::Status MakeErrorStatus(NativeT lhs, NativeT rhs,
                             absl::Span<const int64_t> multi_index) {
  return absl::InvalidArgumentError(absl::StrFormat(
      "first mismatch at array index %s:\n  expected value: %s\n  actual "
      "value:   %s",
      LiteralUtil::MultiIndexAsString(multi_index),
      primitive_util::NativeTypeToString(lhs),
      primitive_util::NativeTypeToString(rhs)));
}

// A recursive function which iterates through every index of expected and
// actual literal and compares their values elementwise. Returns true if all
// elements are equal. Mismatched must either be:
//    - a literal of booleans that has the same shape as expected and actual. In
//      this case, each index in mismatched will be set to true if expected does
//      not equal actual at that index and false if there are equal.
//    - nullptr. In this case, the function will return once any mismatch is
//      found between expected and actual.
template <typename NativeT>
absl::Status Equal(LiteralSlice expected, LiteralSlice actual,
                   absl::Span<int64_t> multi_index, int64_t dimension,
                   Literal* mismatched = nullptr) {
  if (dimension == expected.shape().dimensions_size()) {
    NativeT expected_value = expected.Get<NativeT>(multi_index);
    NativeT actual_value = actual.Get<NativeT>(multi_index);
    bool result =
        CompareEqual<NativeT>(expected_value, actual_value, multi_index);
    if (mismatched) {
      mismatched->Set<bool>(multi_index, !result);
    }
    return result ? absl::OkStatus()
                  : MakeErrorStatus<NativeT>(expected_value, actual_value,
                                             multi_index);
  }

  absl::Status result;
  int64_t upper_bound = expected.shape().dimensions(dimension);
  if (expected.shape().is_dynamic_dimension(dimension)) {
    // If the dimension is dynamic, we only want to check up until the actual
    // dynamic size specified by the literal.
    upper_bound = expected.GetDynamicSize(dimension);
  }

  for (int64_t i = 0; i < upper_bound; ++i) {
    multi_index[dimension] = i;
    if (mismatched != nullptr) {
      result.Update(Equal<NativeT>(expected, actual, multi_index, dimension + 1,
                                   mismatched));
    } else {
      TF_RETURN_IF_ERROR(Equal<NativeT>(expected, actual, multi_index,
                                        dimension + 1, mismatched));
    }
  }
  return result;
}

// Gets the total element count.  For tuples, this is not the count of tuple
// elements, but the sum of elements of each tuple element.
int64_t RecursiveElementCount(const Shape& shape) {
  if (shape.IsTuple()) {
    const int64_t tuple_elements = ShapeUtil::TupleElementCount(shape);
    int64_t total = 0;
    for (int64_t i = 0; i < tuple_elements; ++i) {
      total += RecursiveElementCount(ShapeUtil::GetTupleElementShape(shape, i));
    }
    return total;
  } else if (shape.IsArray()) {
    return ShapeUtil::ElementsIn(shape);
  } else {
    return 0;
  }
}

absl::Status EqualHelper(const LiteralSlice& expected,
                         const LiteralSlice& actual,
                         const ShapeIndex& shape_index,
                         const MiscompareCallback& miscompare_callback) {
  if (expected.shape().is_static() && actual.shape().is_static()) {
    TF_RETURN_IF_ERROR(EqualShapes(expected.shape(), actual.shape()));
  } else {
    TF_RETURN_IF_ERROR(EqualDynamicShapesAndDimensions(expected, actual));
  }

  absl::Status result;
  if (expected.shape().IsTuple()) {
    ShapeIndex next_index = shape_index;
    for (int i = 0; i < ShapeUtil::TupleElementCount(expected.shape()); ++i) {
      next_index.push_back(i);
      absl::Status tuple_result =
          EqualHelper(LiteralSlice(expected, {i}), LiteralSlice(actual, {i}),
                      next_index, miscompare_callback);
      if (miscompare_callback) {
        result.Update(tuple_result);
      } else {
        TF_RETURN_IF_ERROR(tuple_result);
      }
      next_index.pop_back();
    }
  } else {
    std::vector<int64_t> multi_index(expected.shape().dimensions_size(), 0);
    auto index = absl::MakeSpan(multi_index);

    Shape unequal_shape = ShapeUtil::MakeShape(PrimitiveType::PRED,
                                               expected.shape().dimensions());
    Literal miscompared(unequal_shape);
    Literal* miscompared_ptr =
        (miscompare_callback == nullptr ? nullptr : &miscompared);

    primitive_util::PrimitiveTypeSwitch<void>(
        [&](auto primitive_type_constant) -> void {
          if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
            using NativeT =
                primitive_util::NativeTypeOf<primitive_type_constant>;
            result =
                Equal<NativeT>(expected, actual, index, 0, miscompared_ptr);
            return;
          }
          if constexpr (primitive_type_constant == TOKEN) {
            // Tokens have no on-device representation and are trivially equal.
            return;
          }
          LOG(FATAL) << "Unsupported primitive type: "
                     << PrimitiveType_Name(expected.shape().element_type());
        },
        expected.shape().element_type());

    if (!result.ok() && miscompare_callback) {
      miscompare_callback(expected, actual, LiteralSlice(miscompared),
                          shape_index, ErrorBuckets());
    }
  }

  return result;
}

// If result is an error, extend the error message with the expected and actual
// literals.
absl::Status EmitLiteralsInErrorMessage(const absl::Status& result,
                                        const LiteralSlice& expected,
                                        const LiteralSlice& actual) {
  if (result.ok()) {
    return result;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "%s\n\nExpected literal:\n%s\n\nActual literal:\n%s", result.message(),
      ToStringTruncated(expected), ToStringTruncated(actual)));
}

}  // namespace

absl::Status EqualShapes(const Shape& expected, const Shape& actual) {
  if (expected.element_type() != actual.element_type()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "element type mismatch, want: %s got %s",
        ShapeUtil::HumanString(expected), ShapeUtil::HumanString(actual)));
  }
  if (expected.IsTuple()) {
    if (ShapeUtil::TupleElementCount(expected) !=
        ShapeUtil::TupleElementCount(actual)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "want tuple element count: %d got tuple element count: %d",
          ShapeUtil::TupleElementCount(expected),
          ShapeUtil::TupleElementCount(actual)));
    }
    for (int i = 0; i < expected.tuple_shapes_size(); ++i) {
      absl::Status result =
          EqualShapes(expected.tuple_shapes(i), actual.tuple_shapes(i));
      if (!result.ok()) {
        return AppendStatus(result, absl::StrCat("mismatch in tuple index", i));
      }
    }
  } else if (expected.IsArray()) {
    if (expected.rank() != actual.rank()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "want rank of %s got rank of %s", ShapeUtil::HumanString(expected),
          ShapeUtil::HumanString(actual)));
    }
    if (expected.element_type() != actual.element_type()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("mismatch in primitive type %s vs %s",
                          PrimitiveType_Name(expected.element_type()),
                          PrimitiveType_Name(actual.element_type())));
    }
    if (expected.dimensions_size() != actual.dimensions_size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "want dimensions_size %d got dimensions_size %d",
          expected.dimensions_size(), actual.dimensions_size()));
    }
    for (int i = 0; i < expected.dimensions_size(); ++i) {
      if (expected.dimensions(i) != actual.dimensions(i)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "mismatch in dimension #%d expected: %s actual: %s", i,
            ShapeUtil::HumanString(expected), ShapeUtil::HumanString(actual)));
      }
    }
  }
  // Non-array, non-tuple shapes are trivially equivalent.
  return absl::OkStatus();
}

absl::Status EqualDynamicShapesAndDimensions(const LiteralSlice& expected,
                                             const LiteralSlice& actual) {
  TF_RETURN_IF_ERROR(EqualShapes(expected.shape(), actual.shape()));
  return ShapeUtil::ForEachSubshapeWithStatus(
      expected.shape(),
      [&expected, &actual](const Shape& expected_shape,
                           const ShapeIndex& index) -> absl::Status {
        auto actual_shape = ShapeUtil::GetSubshape(actual.shape(), index);
        for (int i = 0; i < expected_shape.dimensions().size(); ++i) {
          if (!expected_shape.is_dynamic_dimension(i) &&
              !actual_shape.is_dynamic_dimension(i)) {
            // We're only interested in dynamic dimensions.
            continue;
          }
          if (expected_shape.is_dynamic_dimension(i) &&
              !actual_shape.is_dynamic_dimension(i)) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "mismatch at dimension %d. the expected shape %s is dynamic "
                "while "
                "the actual shape %s is not.",
                i, ShapeUtil::HumanString(expected.shape()),
                ShapeUtil::HumanString(actual.shape())));
          }
          if (!expected_shape.is_dynamic_dimension(i) &&
              actual_shape.is_dynamic_dimension(i)) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "mismatch at dimension %d. the expected shape %s is not "
                "dynamic "
                "while the actual shape %s is dynamic.",
                i, ShapeUtil::HumanString(expected.shape()),
                ShapeUtil::HumanString(actual.shape())));
          }
          // Both dimensions are dynamic. Check that they are equal.
          int64_t expected_dynamic_size = expected.GetDynamicSize(i, index);
          int64_t actual_dynamic_size = actual.GetDynamicSize(i, index);
          if (expected_dynamic_size != actual_dynamic_size) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "mismatch at dimension %d. The expected dynamic size does not "
                "match "
                "the actual dynamic size. %d vs. %d",
                i, expected_dynamic_size, actual_dynamic_size));
          }
        }

        return absl::OkStatus();
      });
}

absl::Status Equal(const LiteralSlice& expected, const LiteralSlice& actual) {
  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "expected:";
    ZKX_LOG_LINES(INFO, expected.ToString());
    LOG(INFO) << "actual:";
    ZKX_LOG_LINES(INFO, actual.ToString());
  }
  absl::Status result = EqualHelper(expected, actual, {}, nullptr);
  return EmitLiteralsInErrorMessage(result, expected, actual);
}

std::string ToStringTruncated(const LiteralSlice& literal) {
  return RecursiveElementCount(literal.shape()) < 1000
             ? literal.ToString()
             : "[TRUNCATED, Literal with more than 1000 values]";
}

}  // namespace zkx::literal_comparison
