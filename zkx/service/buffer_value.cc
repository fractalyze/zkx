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

#include "zkx/service/buffer_value.h"

#include <ostream>

namespace zkx {

BufferValue::BufferValue(HloInstruction* instruction, const ShapeIndex& index,
                         Id id)
    : id_(id) {
  const Shape& shape = ShapeUtil::GetSubshape(instruction->shape(), index);
  is_array_ = shape.IsArray();
  is_tuple_ = shape.IsTuple();
}

std::ostream& operator<<(std::ostream& out, const BufferValue& buffer) {
  out << buffer.ToString();
  return out;
}

}  // namespace zkx
