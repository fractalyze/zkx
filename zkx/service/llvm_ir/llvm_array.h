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

#ifndef ZKX_SERVICE_LLVM_IR_LLVM_ARRAY_H_
#define ZKX_SERVICE_LLVM_IR_LLVM_ARRAY_H_

#include <map>

#include "absl/log/check.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "zkx/layout.h"
#include "zkx/map_util.h"
#include "zkx/shape.h"

namespace zkx::llvm_ir {

// LlvmArray represents a ZKX array at the LLVM IR level. This class
// encapsulates a base pointer to the buffer holding the array (as an LLVM
// Value) and the shape of the array. The class includes methods for emitting
// LLVM IR sequences which access elements of the array at a multidimensional
// index (eg, [x, y, z] in a 3-dimensional array). Arbitrary shape and layouts
// are supported.
class LlvmArray {
 public:
  // Default constructor. Constructs an LlvmArray in a null status.
  LlvmArray() : base_ptr_(nullptr) {}

  // Construct an LlvmArray with the given base pointer, pointee type, and
  // shape. base_ptr is a pointer type pointing to the first element(lowest
  // address) of the array.
  //
  // For packed arrays, base_ptr points to packed memory with the correct number
  // of elements when unpacked. pointee_type should be an iN array in this case,
  // and reads and writes will return or take in iN values. LlvmArray internally
  // reads or writes i8 values, by treating base_ptr as an i8 array and
  // masking/shifting on the fly. LlvmArray does not directly read/write iN
  // values, since arrays of iN values in LLVM are not packed (every element of
  // an LLVM IR array must have unique address).
  LlvmArray(llvm::Value* base_ptr, llvm::Type* pointee_type, Shape shape);

  // Default implementations of copying and moving.
  LlvmArray(LlvmArray&& other) noexcept = default;
  LlvmArray(const LlvmArray& other) = default;
  LlvmArray& operator=(LlvmArray&& other) noexcept = default;
  LlvmArray& operator=(const LlvmArray& other) = default;

  llvm::Value* GetBasePointer() const { return base_ptr_; }
  llvm::Type* GetBasePointeeType() const { return pointee_type_; }
  llvm::Type* GetElementLlvmType() const { return element_type_; }

  const Shape& GetShape() const { return shape_; }

  void AddAliasScopeMetadata(llvm::MDNode* alias_scope) {
    CHECK_NE(alias_scope, nullptr);
    AddMetadata(llvm::LLVMContext::MD_alias_scope, alias_scope);
  }

  void AddNoaliasMetadata(llvm::MDNode* noalias) {
    CHECK_NE(noalias, nullptr);
    AddMetadata(llvm::LLVMContext::MD_noalias, noalias);
  }

  // Promises LLVM that the data pointed to by this LlvmArray never changes
  // after it's first loaded.
  //
  // The temporal scope of this promise is the "whole program" from LLVM's point
  // of view, but how this translates to HLOs differs between backends.
  //
  // In the single-threaded CPU backend, we emit one function that
  // runs all the HLOs in sequence, so the whole program is the whole HLO
  // module.
  //
  // In the GPU backend, we emit one GPU kernel per top-level HLO (i.e. per HLO
  // in the entry computation). From LLVM's perspective, launching a new kernel
  // is like launching a new program, and so the whole program is one top-level
  // HLO. Since the scope of the promise is smaller than in the CPU backend, we
  // can mark more things as invariant in the GPU backend.
  //
  // Marking loads as invariant is particularly helpful on GPUs because
  // invariant loads can be lowered to PTX ld.global.nc (equivalent to CUDA's
  // __ldg intrinsic). These loads use a special cache, and can be
  // significantly faster than regular loads.
  void MarkInvariantOverWholeProgram(llvm::LLVMContext* context) {
    if (is_invariant_) {
      return;
    }
    is_invariant_ = true;
    AddMetadata(llvm::LLVMContext::MD_invariant_load,
                llvm::MDNode::get(*context, {}));
  }

  const std::map<int, llvm::MDNode*>& metadata() const { return metadata_; }

 private:
  // Add the specified LLVM IR metadata to loads/stores associated with this
  // LlvmArray.
  void AddMetadata(int kind, llvm::MDNode* md) {
    InsertOrDie(&metadata_, kind, md);
  }

  // Address of the base of the array as an LLVM Value.
  llvm::Value* base_ptr_;

  // The pointee type of base_ptr_;
  llvm::Type* pointee_type_;

  // The LLVM type of the elements in the array.
  llvm::Type* element_type_;

  // Shape of the ZKX array.
  Shape shape_;

  // The list of key/value pairs used when attaching metadata to emitted
  // loads/stores for this array. They keys are the metadata kinds and the
  // values are the metadata nodes.
  std::map<int, llvm::MDNode*> metadata_;

  bool is_invariant_ = false;
};

}  // namespace zkx::llvm_ir

#endif  // ZKX_SERVICE_LLVM_IR_IR_ARRAY_H_
