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

#ifndef ZKX_BASE_FLAG_FLAG_FORWARD_H_
#define ZKX_BASE_FLAG_FLAG_FORWARD_H_

#include <stdint.h>

#include <string>

namespace zkx::base {

template <typename T>
class Flag;

using BoolFlag = Flag<bool>;
using IntFlag = Flag<int>;
using Uint8Flag = Flag<uint8_t>;
using Int8Flag = Flag<int8_t>;
using Uint16Flag = Flag<uint16_t>;
using Int16Flag = Flag<int16_t>;
using Uint32Flag = Flag<uint32_t>;
using Int32Flag = Flag<int32_t>;
using Uint64Flag = Flag<uint64_t>;
using Int64Flag = Flag<int64_t>;
using FloatFlag = Flag<float>;
using DoubleFlag = Flag<double>;
using StringFlag = Flag<std::string>;

template <typename T>
class ChoicesFlag;

using StringChoicesFlag = ChoicesFlag<std::string>;

template <typename T>
class RangeFlag;

using Uint8RangeFlag = RangeFlag<uint8_t>;
using Int8RangeFlag = RangeFlag<int8_t>;
using Uint16RangeFlag = RangeFlag<uint16_t>;
using Int16RangeFlag = RangeFlag<int16_t>;
using Uint32RangeFlag = RangeFlag<uint32_t>;
using Int32RangeFlag = RangeFlag<int32_t>;
using Uint64RangeFlag = RangeFlag<uint64_t>;
using Int64RangeFlag = RangeFlag<int64_t>;
using FloatRangeFlag = RangeFlag<float>;
using DoubleRangeFlag = RangeFlag<double>;

}  // namespace zkx::base

#endif  // ZKX_BASE_FLAG_FLAG_FORWARD_H_
