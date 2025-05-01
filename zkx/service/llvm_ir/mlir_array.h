#ifndef ZKX_SERVICE_LLVM_IR_MLIR_ARRAY_H_
#define ZKX_SERVICE_LLVM_IR_MLIR_ARRAY_H_

#include <stdint.h>

#include <vector>

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "zkx/codegen/emitter_loc_op_builder.h"
#include "zkx/shape.h"
#include "zkx/shape_util.h"

namespace zkx::llvm_ir {

class MlirArray {
 public:
  class Index {
   public:
    // Constructs an index for a scalar shape.
    explicit Index(mlir::Type index_ty) : index_type_(index_ty) {
      CHECK(index_ty.isInteger());
    }

    const std::vector<mlir::Value>& multidim() const { return multidim_; }
    const std::vector<int64_t>& dims() const { return dims_; }
    mlir::Value linear() const { return linear_; }

    size_t size() const { return multidim().size(); }

    mlir::Value operator[](size_t i) const { return multidim()[i]; }

    bool LinearValidOnShape(const Shape& a) const;

    static bool ShapeIsCompatible(const Shape& a, const Shape& b);

    bool ShapeIsCompatible(const Shape& a) const {
      return ShapeIsCompatible(a, AsShapeWithType(a.element_type()));
    }

    Shape AsShapeWithType(PrimitiveType element_type) const {
      return ShapeUtil::MakeShapeWithDenseLayout(element_type, dims_,
                                                 layout_.minor_to_major());
    }

    mlir::Type GetType() const { return index_type_; }

   private:
    std::vector<mlir::Value> multidim_;

    // These values are purely for efficiency; `multidim_` is enough to find the
    // element at a given `Index`, but if a loop is emitted with a linear index
    // space, that linear index can be saved in `linear_`, and the layout and
    // dimensions of the shape the loop was emitted for in `layout_` and
    // `dims_`, and if the `Index` is used in another array, and its layout and
    // dimensions match, the linear index can be used, sparing the cost of
    // computing `multidim_`, which LLVM DCE could potentially so delete.
    // Modifying `multidim_` after construction nullifies `linear_`, lest it
    // be used wrongly, as it would be valid no more.
    // If a loop is emitted with a multidimensional index space, `linear_` would
    // be null and `layout_` and `dims_` would be ignored.
    mlir::Value linear_;
    Layout layout_;
    std::vector<int64_t> dims_;

    mlir::Type index_type_;
  };

  // Construct a MlirArray with the given base pointer, pointee type, and shape.
  // base_ptr is a pointer type pointing to the first element(lowest address)
  // of the array.
  //
  // For packed arrays, `base_ptr` points to packed memory with the correct
  // number of elements when unpacked. pointee_type should be an iN array in
  // this case, and reads and writes will return or take in iN values. MlirArray
  // internally reads or writes i8 values, by treating base_ptr as an i8 array
  // and masking/shifting on the fly. MlirArray does not directly read/write iN
  // values, since arrays of iN values in LLVM are not packed (every element of
  // an LLVM IR array must have unique address).
  MlirArray(mlir::Value base_ptr, mlir::Type pointee_type, Shape shape);

  // Default implementations of copying and moving.
  MlirArray(MlirArray&& other) noexcept = default;
  MlirArray(const MlirArray& other) = default;
  MlirArray& operator=(MlirArray&& other) noexcept = default;
  MlirArray& operator=(const MlirArray& other) = default;

  const Shape& GetShape() const { return shape_; }

  // Emit a sequence of instructions to compute the address of the element in
  // the given array at the given index. Returns the address of the element as
  // an MLIR Value.
  //
  // The optional name is useful for debugging when looking at
  // the emitted MLIR IR.
  //
  // `bit_offset` contains the offset of the element inside the address.
  mlir::Value EmitArrayElementAddress(const Index& index,
                                      EmitterLocOpBuilder& b,
                                      std::string_view name = "",
                                      bool use_linear_index = true,
                                      mlir::Value* bit_offset = nullptr) const;

  // Emit IR to read an array element at the given index. Returns the read
  // result (effectively, a Value loaded from memory). This method seamlessly
  // handles scalar shapes by broadcasting their value to all indices (index is
  // ignored).
  //
  // The optional name is useful for debugging when looking at
  // the emitted MLIR IR.
  // `use_linear_index` can be used to specify whether the linear index (if
  // available) or the multi-dimensional index should be used.
  mlir::Value EmitReadArrayElement(const Index& index, EmitterLocOpBuilder& b,
                                   std::string_view name = "",
                                   bool use_linear_index = true) const;

  // Emit IR to write the given value to the array element at the given index.
  // `use_linear_index` can be used to specify whether the linear index (if
  // available) or the multi-dimensional index should be used.
  //
  // For packed arrays, only part of the byte in the array is written. First
  // the appropriate byte is read from the array, then a subset of bits are
  // modified and written back. To avoid race conditions, the caller must ensure
  // that the different values within a byte are not written to in parallel.
  void EmitWriteArrayElement(const Index& index, mlir::Value value,
                             EmitterLocOpBuilder& b,
                             bool use_linear_index = true) const;

 private:
  // Address of the base of the array as an MLIR Value.
  mlir::Value base_ptr_;

  // The pointee type of `base_ptr_`;
  mlir::Type pointee_type_;

  // The LLVM type of the elements in the array.
  mlir::Type element_type_;

  // Shape of the ZKX array.
  Shape shape_;
};

}  // namespace zkx::llvm_ir

#endif  // ZKX_SERVICE_LLVM_IR_MLIR_ARRAY_H_
