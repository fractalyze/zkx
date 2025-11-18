// Copyright (c) 2020 The Console Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE.console file.

/* Copyright 2025 The ZKX Authors.

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

#ifndef ZKX_BASE_FLAG_FLAG_VALUE_TRAITS_H_
#define ZKX_BASE_FLAG_FLAG_VALUE_TRAITS_H_

#include <stdint.h>

#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/substitute.h"

#include "xla/tsl/platform/errors.h"

namespace zkx::base {

template <typename T, typename SFINAE = void>
class FlagValueTraits;

template <typename T>
class FlagValueTraits<
    T, std::enable_if_t<std::is_integral<T>::value &&
                        std::is_signed<T>::value && (sizeof(T) <= 32)>> {
 public:
  static absl::Status ParseValue(std::string_view input, T* value) {
    int value_tmp;
    if (!absl::SimpleAtoi(input, &value_tmp)) {
      return absl::InvalidArgumentError(
          absl::Substitute("failed to convert int (\"$0\")", input));
    }
    if (value_tmp <= std::numeric_limits<T>::max() &&
        value_tmp >= std::numeric_limits<T>::min()) {
      *value = static_cast<T>(value_tmp);
      return absl::OkStatus();
    }
    return absl::OutOfRangeError(
        absl::Substitute("$0 is out of its range", input));
  }
};

template <typename T>
class FlagValueTraits<
    T,
    std::enable_if_t<std::is_integral<T>::value && !std::is_signed<T>::value &&
                     !std::is_same<T, bool>::value && (sizeof(T) <= 32)>> {
 public:
  static absl::Status ParseValue(std::string_view input, T* value) {
    unsigned value_tmp;
    if (!absl::SimpleAtoi(input, &value_tmp)) {
      return absl::InvalidArgumentError(
          absl::Substitute("failed to convert unsigned int (\"$0\")", input));
    }
    if (value_tmp <= std::numeric_limits<T>::max()) {
      *value = static_cast<T>(value_tmp);
      return absl::OkStatus();
    }
    return absl::OutOfRangeError(
        absl::Substitute("$0 is out of its range", input));
  }
};

template <>
class FlagValueTraits<float> {
 public:
  static absl::Status ParseValue(std::string_view input, float* value) {
    if (!absl::SimpleAtof(input, value)) {
      return absl::InvalidArgumentError(
          absl::Substitute("failed to convert to float (\"$0\")", input));
    }
    return absl::OkStatus();
  }
};

template <>
class FlagValueTraits<double> {
 public:
  static absl::Status ParseValue(std::string_view input, double* value) {
    if (!absl::SimpleAtod(input, value)) {
      return absl::InvalidArgumentError(
          absl::Substitute("failed to convert to float (\"$0\")", input));
    }
    return absl::OkStatus();
  }
};

template <>
class FlagValueTraits<int64_t> {
 public:
  static absl::Status ParseValue(std::string_view input, int64_t* value) {
    if (!absl::SimpleAtoi(input, value)) {
      return absl::InvalidArgumentError(
          absl::Substitute("failed to convert to int64_t (\"$0\")", input));
    }
    return absl::OkStatus();
  }
};

template <>
class FlagValueTraits<uint64_t> {
 public:
  static absl::Status ParseValue(std::string_view input, uint64_t* value) {
    if (!absl::SimpleAtoi(input, value)) {
      return absl::InvalidArgumentError(
          absl::Substitute("failed to convert to uint64_t (\"$0\")", input));
    }
    return absl::OkStatus();
  }
};

template <>
class FlagValueTraits<bool> {
 public:
  static absl::Status ParseValue(std::string_view input, bool* value) {
    *value = true;
    return absl::OkStatus();
  }
};

template <>
class FlagValueTraits<std::string> {
 public:
  static absl::Status ParseValue(std::string_view input, std::string* value) {
    if (input.length() == 0) {
      return absl::InvalidArgumentError("input is empty");
    }
    *value = std::string(input);
    return absl::OkStatus();
  }
};

template <typename T>
class FlagValueTraits<std::vector<T>> {
 public:
  static absl::Status ParseValue(std::string_view input,
                                 std::vector<T>* value) {
    T element;
    TF_RETURN_IF_ERROR(FlagValueTraits<T>::ParseValue(input, &element));
    value->push_back(std::move(element));
    return absl::OkStatus();
  }
};

template <typename T>
class FlagValueTraits<std::set<T>> {
 public:
  static absl::Status ParseValue(std::string_view input, std::set<T>* value) {
    T element;
    TF_RETURN_IF_ERROR(FlagValueTraits<T>::ParseValue(input, &element));
    value->insert(std::move(element));
    return absl::OkStatus();
  }
};

}  // namespace zkx::base

#endif  // ZKX_BASE_FLAG_FLAG_VALUE_TRAITS_H_
