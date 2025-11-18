// Copyright 2022 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE.chromium file.

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

#ifndef ZKX_BASE_TYPES_CXX20_IS_BOUNDED_ARRAY_H_
#define ZKX_BASE_TYPES_CXX20_IS_BOUNDED_ARRAY_H_

#include <stddef.h>

#include <type_traits>

namespace zkx::base {

// Implementation of C++20's std::is_bounded_array.
//
// References:
// - https://en.cppreference.com/w/cpp/types/is_bounded_array
template <typename T>
struct is_bounded_array : std::false_type {};

template <typename T, size_t N>
struct is_bounded_array<T[N]> : std::true_type {};

template <typename T>
inline constexpr bool is_bounded_array_v = is_bounded_array<T>::value;

}  // namespace zkx::base

#endif  // ZKX_BASE_TYPES_CXX20_IS_BOUNDED_ARRAY_H_
