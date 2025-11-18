/* Copyright 2021 The OpenXLA Authors.
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

#ifndef ZKX_MLIR_HLO_MHLO_UTILS_TYPE_CONVERSION_H_
#define ZKX_MLIR_HLO_MHLO_UTILS_TYPE_CONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::mhlo {

// Type converter to use as part of lowerings from dialects that carry signs
// in their types to those that are signless.
class RemoveSignTypeConverter : public TypeConverter {
 public:
  RemoveSignTypeConverter();
};

}  // namespace mlir::mhlo

#endif  // ZKX_MLIR_HLO_MHLO_UTILS_TYPE_CONVERSION_H_
