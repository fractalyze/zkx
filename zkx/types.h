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

#ifndef ZKX_TYPES_H_
#define ZKX_TYPES_H_

#include <limits>
#include <type_traits>

#include "zk_dtypes/include/intn.h"

namespace zkx {

template <typename T>
struct is_specialized_integral
    : std::bool_constant<std::numeric_limits<T>::is_specialized &&
                         std::numeric_limits<T>::is_integer> {};

template <typename T>
inline constexpr bool is_specialized_integral_v =
    is_specialized_integral<T>::value;

using u1 = ::zk_dtypes::uint1;
using s1 = ::zk_dtypes::int1;
using u2 = ::zk_dtypes::uint2;
using s2 = ::zk_dtypes::int2;
using u4 = ::zk_dtypes::uint4;
using s4 = ::zk_dtypes::int4;

}  // namespace zkx

#endif  // ZKX_TYPES_H_
