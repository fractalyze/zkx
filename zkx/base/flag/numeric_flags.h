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

#ifndef ZKX_BASE_FLAG_NUMERIC_FLAGS_H_
#define ZKX_BASE_FLAG_NUMERIC_FLAGS_H_

#include "xla/tsl/platform/errors.h"
#include "zkx/base/flag/flag_value_traits.h"

namespace zkx::base {

template <typename T,
          typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
absl::Status ParsePositiveValue(std::string_view arg, T* value) {
  T n;
  TF_RETURN_IF_ERROR(FlagValueTraits<T>::ParseValue(arg, &n));
  if (n > 0) {
    *value = n;
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::Substitute("$0 is not positive", arg));
}

}  // namespace zkx::base

#endif  // ZKX_BASE_FLAG_NUMERIC_FLAGS_H_
