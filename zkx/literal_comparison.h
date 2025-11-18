/* Copyright 2018 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

// Library for comparing literals without taking a dependency on testing
// libraries.

#ifndef ZKX_LITERAL_COMPARISON_H_
#define ZKX_LITERAL_COMPARISON_H_

#include "absl/status/status.h"

#include "zkx/literal.h"

namespace zkx::literal_comparison {

// Returns ok if the given shapes have the same rank, dimension sizes, and
// primitive types.
absl::Status EqualShapes(const Shape& expected, const Shape& actual);

// Returns ok if the given literals share identical dynamic shapes and
// dimension sizes.
absl::Status EqualDynamicShapesAndDimensions(const LiteralSlice& expected,
                                             const LiteralSlice& actual);

// Returns ok if the expected and actual literals are (bitwise) equal for all
// elements in the literal. Also, asserts that the rank, dimensions sizes, and
// primitive type are equal.
absl::Status Equal(const LiteralSlice& expected, const LiteralSlice& actual);

// Structure that contains the distribution of absolute and relative errors,
// bucketized into five buckets: [0.0001, 0.001, 0.01, 0.1, 1].
// Useful to understand the distribution of errors and set the permissible
// error bounds in an ErrorSpec.
struct ErrorBuckets {
  explicit ErrorBuckets(const std::vector<int64_t>& absolute_error_buckets = {},
                        const std::vector<int64_t>& rel_error_buckets = {})
      : abs_error_buckets(absolute_error_buckets),
        rel_error_buckets(rel_error_buckets) {}

  const std::vector<int64_t> abs_error_buckets;
  const std::vector<int64_t> rel_error_buckets;
};

using MiscompareCallback = std::function<void(
    const LiteralSlice& expected, const LiteralSlice& actual,
    const LiteralSlice& mismatches, const ShapeIndex& shape_index,
    const ErrorBuckets& error_buckets)>;

// Calling ToString on a literal with over 100 million elements takes around
// 3 minutes.  The utility of printing a literal with >1000 elements is
// questionable, especially when writing the Literal proto to disk is orders
// of magnitude faster.
std::string ToStringTruncated(const LiteralSlice& literal);

}  // namespace zkx::literal_comparison

#endif  // ZKX_LITERAL_COMPARISON_H_
