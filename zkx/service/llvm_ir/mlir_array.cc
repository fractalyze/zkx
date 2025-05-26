#include "zkx/service/llvm_ir/mlir_array.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "xla/tsl/platform/status.h"
#include "zkx/layout_util.h"
#include "zkx/shape_util.h"

namespace zkx::llvm_ir {

MlirArray::Index::Index(absl::Span<mlir::Value const> multidim,
                        absl::Span<int64_t const> dimensions,
                        mlir::Type index_type)
    : Index(multidim, ShapeUtil::MakeShape(/*arbitrary*/ PRED, dimensions),
            index_type) {}

MlirArray::Index::Index(absl::Span<mlir::Value const> multidim,
                        const Shape& shape, mlir::Type index_type)
    : multidim_(multidim.begin(), multidim.end()),
      linear_(nullptr),
      layout_(shape.layout()),
      dims_(shape.dimensions().begin(), shape.dimensions().end()),
      index_type_(index_type) {
  CHECK_EQ(shape.dimensions_size(), multidim.size());
  for (const mlir::Value dim : multidim) {
    CHECK(dim);
  }
  CHECK(LayoutUtil::HasLayout(shape))
      << "Shape " << ShapeUtil::HumanStringWithLayout(shape)
      << " should have a layout.";
}

MlirArray::MlirArray(mlir::Value base_ptr, mlir::Type pointee_type, Shape shape)
    : base_ptr_(base_ptr),
      pointee_type_(pointee_type),
      shape_(std::move(shape)) {
  TF_CHECK_OK(ShapeUtil::ValidateShape(shape_));
  CHECK(mlir::isa<mlir::LLVM::LLVMPointerType>(base_ptr_.getType()));
  element_type_ = pointee_type;
  int depth = 0;
  while (mlir::LLVM::LLVMArrayType array_type =
             mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(element_type_)) {
    element_type_ = array_type.getElementType();
    ++depth;
  }

  if (!shape_.IsArray() || ShapeUtil::IsScalar(shape_)) {
    DCHECK(depth == 1 || depth == 0) << depth;
  } else {
    DCHECK_EQ(depth, shape_.rank()) << shape.ShortDebugString();
  }
}

// Returns whether the given linear index is valid on the given shape.
bool MlirArray::Index::LinearValidOnShape(const Shape& a) const {
  auto b = ShapeUtil::MakeShape(a.element_type(), dims_);
  *b.mutable_layout() = layout_;
  return linear_ != nullptr &&
         ShapeUtil::ElementsIn(a) == ShapeUtil::ElementsIn(b) &&
         ShapeUtil::ReshapeIsBitcast(a, b);
}

mlir::Value MlirArray::EmitArrayElementAddress(const Index& index,
                                               EmitterLocOpBuilder& b,
                                               std::string_view name,
                                               bool use_linear_index,
                                               mlir::Value* bit_offset) const {
  if (ShapeUtil::IsScalar(shape_)) {
    if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
      LOG(FATAL) << "Not implemented";
    }
    // Special handling of scalars: a scalar pretends to have the same value for
    // every index, thus effectively implementing broadcasting of its value
    // over higher-rank arrays.
    return base_ptr_;
  }
  CHECK_EQ(index.size(), shape_.rank());
  CHECK(index.ShapeIsCompatible(shape_))
      << "Shape " << index.AsShapeWithType(shape_.element_type()).ToString(true)
      << " is not compatible with " << shape_.ToString(true);

  if (use_linear_index && index.LinearValidOnShape(shape_)) {
    LOG(FATAL) << "Not implemented";
  }

  if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
    LOG(FATAL) << "Not implemented";
  }
  std::vector<mlir::Value> actual_index;
  actual_index.reserve(index.size());
  for (int64_t i = 0; i < index.size(); ++i) {
    // When dimension i is of size 1, LLVM optimization is able to replace
    // index[i] with 0. However, setting index[i] to 0 here still allows LLVM to
    // produce better code in some cases.
    auto dim = shape_.dimensions(i);
    actual_index.push_back(
        dim == 1 ? b.create<mlir::LLVM::ConstantOp>(index[i].getType(), 0)
                 : index[i]);
  }

  // `base_ptr_` has the type of "<ir_type_for_its_shape>*"
  // (e.g. [3 x [2 x i32]]*). Therefore, the address of the indexed element
  // should be computed by
  //
  //   getelementptr base_ptr_, 0, most major index, ..., most minor index
  CHECK_GT(index.size(), 0);
  std::vector<mlir::Value> gep_indices(
      1, b.create<mlir::LLVM::ConstantOp>(index[0].getType(), 0));
  gep_indices.reserve(shape_.rank() + 1);
  for (int64_t i = 0; i < shape_.rank(); ++i) {
    int64_t dimension = LayoutUtil::Major(shape_.layout(), i);
    gep_indices.push_back(actual_index[dimension]);
  }
  return b.create<mlir::LLVM::GEPOp>(b.getType<mlir::LLVM::LLVMPointerType>(),
                                     pointee_type_, base_ptr_, gep_indices,
                                     /*inbounds=*/true);
}

mlir::Value MlirArray::EmitReadArrayElement(const Index& index,
                                            EmitterLocOpBuilder& b,
                                            std::string_view name,
                                            bool use_linear_index) const {
  mlir::Value bit_offset;
  mlir::Value element_address =
      EmitArrayElementAddress(index, b, name, use_linear_index, &bit_offset);
  mlir::Type load_type =
      primitive_util::IsSubByteNonPredType(shape_.element_type())
          ? b.getI8Type()
          : element_type_;
  // TODO(chokobole): add name to `load`.
  mlir::Value load = b.create<mlir::LLVM::LoadOp>(load_type, element_address);
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: AnnotateLoadStoreInstructionWithMetadata
  // clang-format on
  // AnnotateLoadStoreInstructionWithMetadata(load);
  mlir::Value elem = load;
  if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
    // TODO(chokobole): Implement this.
    LOG(FATAL) << "Not implemented";
  }
  return elem;
}

void MlirArray::EmitWriteArrayElement(const Index& index, mlir::Value value,
                                      EmitterLocOpBuilder& b,
                                      bool use_linear_index) const {
  mlir::Value bit_offset;
  mlir::Value element_address =
      EmitArrayElementAddress(index, b, "", use_linear_index, &bit_offset);
  if (primitive_util::IsSubByteNonPredType(shape_.element_type())) {
    // TODO(chokobole): Implement this.
    LOG(FATAL) << "Not implemented";
  }
  b.create<mlir::LLVM::StoreOp>(value, element_address);
  // clang-format off
  // TODO(chokobole): Uncomment this. Dependency: AnnotateLoadStoreInstructionWithMetadata
  // clang-format on
  // AnnotateLoadStoreInstructionWithMetadata(store);
}

// static
bool MlirArray::Index::ShapeIsCompatible(const Shape& a, const Shape& b) {
  // Compute strides for two sides of the comparison. Sometimes different shapes
  // give the same strides:
  //   [10, 20, 30, 1]{3,2,1,0} vs [10, 20, 1, 30]{3,2,1,0}
  // which should be considered compatible.
  const auto get_strides = [](const Shape& shape) {
    int rank = shape.dimensions().size();
    int64_t stride = 1;
    std::vector<int64_t> strides;
    for (int i = 0; i < rank; i++) {
      auto dim = shape.dimensions(shape.layout().minor_to_major(i));
      if (dim != 1) {
        stride *= dim;
        strides.push_back(stride);
      }
    }
    return strides;
  };

  return get_strides(a) == get_strides(b);
}

}  // namespace zkx::llvm_ir
