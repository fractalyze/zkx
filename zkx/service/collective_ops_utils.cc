/* Copyright 2019 The OpenXLA Authors.

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

#include "zkx/service/collective_ops_utils.h"

namespace zkx {

std::string_view ReductionKindToString(ReductionKind reduction_kind) {
  switch (reduction_kind) {
    case ReductionKind::kSum:
      return "sum";
    case ReductionKind::kProduct:
      return "prod";
    case ReductionKind::kMin:
      return "min";
    case ReductionKind::kMax:
      return "max";
  }
}

}  // namespace zkx
