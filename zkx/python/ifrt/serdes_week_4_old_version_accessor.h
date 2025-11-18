/* Copyright 2025 The OpenXLA Authors.
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

#ifndef ZKX_PYTHON_IFRT_SERDES_WEEK_4_OLD_VERSION_ACCESSOR_H_
#define ZKX_PYTHON_IFRT_SERDES_WEEK_4_OLD_VERSION_ACCESSOR_H_

#include "zkx/python/ifrt/serdes_version.h"

namespace zkx::ifrt {

// Accessor for `SerDesVersion` that allows getting the 4-week-old version.
// Typically used for the layers that produce an artifact loaded by a
// potentially old binary that does not yet support the latest SerDes version.
class SerDesWeek4OldVersionAccessor {
 public:
  static SerDesVersion Get() { return SerDesVersion::week_4_old(); }
};

}  // namespace zkx::ifrt

#endif  // ZKX_PYTHON_IFRT_SERDES_WEEK_4_OLD_VERSION_ACCESSOR_H_
