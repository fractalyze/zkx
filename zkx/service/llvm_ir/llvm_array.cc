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

#include "zkx/service/llvm_ir/llvm_array.h"

#include <utility>

#include "llvm/IR/DerivedTypes.h"

#include "xla/tsl/platform/status.h"
#include "zkx/shape_util.h"

namespace zkx::llvm_ir {

LlvmArray::LlvmArray(llvm::Value* base_ptr, llvm::Type* pointee_type,
                     Shape shape)
    : base_ptr_(base_ptr),
      pointee_type_(pointee_type),
      shape_(std::move(shape)) {
  TF_CHECK_OK(ShapeUtil::ValidateShape(shape));
  CHECK(base_ptr_->getType()->isPointerTy());
  int depth = 0;
  element_type_ = pointee_type;
  while (llvm::ArrayType* array_type =
             llvm::dyn_cast<llvm::ArrayType>(element_type_)) {
    element_type_ = array_type->getElementType();
    ++depth;
  }

  if (!shape_.IsArray() || ShapeUtil::IsScalar(shape_)) {
    DCHECK(depth == 1 || depth == 0) << depth;
  } else {
    DCHECK_EQ(depth, shape_.rank()) << shape.ShortDebugString();
  }
}

}  // namespace zkx::llvm_ir
