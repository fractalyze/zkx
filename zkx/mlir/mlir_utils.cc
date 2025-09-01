#include "zkx/mlir/mlir_utils.h"

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

#include "zkx/layout_util.h"
#include "zkx/math/elliptic_curves/bn/bn254/fr.h"
#include "zkx/math/elliptic_curves/bn/bn254/g1.h"
#include "zkx/math/elliptic_curves/bn/bn254/g2.h"
#include "zkx/primitive_util.h"

namespace zkx::mlir_utils {

mlir::Type PrimitiveTypeToMlirType(PrimitiveType element_type,
                                   mlir::MLIRContext* context,
                                   bool use_montgomery) {
  switch (element_type) {
    case PRED:
    case S1:
    case U1:
    case S2:
    case U2:
    case S4:
    case U4:
    case S8:
    case U8:
    case S16:
    case U16:
    case S32:
    case U32:
    case S64:
    case U64:
      return mlir::IntegerType::get(context,
                                    primitive_util::BitWidth(element_type));
    // TODO(chokobole): For Tuple, see the comments in
    // ShapeToMlirMemRefType().
    case TUPLE:
    // An Opaque is like a void*, use i8*.
    case OPAQUE_TYPE:
      return mlir::MemRefType::get({1}, mlir::IntegerType::get(context, 8));
    case TOKEN:
      // Tokens do not have a physical representation, but the compiler needs
      // some placeholder type, so use int8_t*.
      return mlir::MemRefType::get({1}, mlir::IntegerType::get(context, 8));
    case BN254_SCALAR:
      return GetMlirPrimeFieldType<math::bn254::Fr>(context, use_montgomery);
    case BN254_G1_AFFINE:
      return GetMlirAffinePointType<math::bn254::G1AffinePoint>(context,
                                                                use_montgomery);
    case BN254_G1_JACOBIAN:
      return GetMlirJacobianPointType<math::bn254::G1JacobianPoint>(
          context, use_montgomery);
    case BN254_G1_XYZZ:
      return GetMlirPointXyzzType<math::bn254::G1PointXyzz>(context,
                                                            use_montgomery);
    case BN254_G2_AFFINE:
      return GetMlirAffinePointType<math::bn254::G2AffinePoint>(context,
                                                                use_montgomery);
    case BN254_G2_JACOBIAN:
      return GetMlirJacobianPointType<math::bn254::G2JacobianPoint>(
          context, use_montgomery);
    case BN254_G2_XYZZ:
      return GetMlirPointXyzzType<math::bn254::G2PointXyzz>(context,
                                                            use_montgomery);
    default:
      LOG(FATAL) << "unsupported type " << element_type;
  }
}

mlir::Type PrimitiveTypeToMlirTypeWithSign(PrimitiveType element_type,
                                           mlir::MLIRContext* context,
                                           bool use_montgomery) {
  if (element_type == PRED) {
    return mlir::IntegerType::get(context, 1);
  } else if (primitive_util::IsIntegralType(element_type)) {
    return mlir::IntegerType::get(
        context,
        /*width=*/primitive_util::BitWidth(element_type),
        /*signed=*/
        primitive_util::IsUnsignedIntegralType(element_type)
            ? mlir::IntegerType::Unsigned
            : mlir::IntegerType::Signless);
  }
  // Delegate to the other function for non-integer and signed integer
  // types.
  return PrimitiveTypeToMlirType(element_type, context, use_montgomery);
}

mlir::MemRefType ShapeToMlirMemRefType(const Shape& shape,
                                       mlir::MLIRContext* context) {
  CHECK(shape.IsArray());
  CHECK(shape.is_static())
      << "ShapeToMlirMemRefType only supports static shapes.";
  const Layout& layout = shape.layout();
  mlir::Type element_type = PrimitiveTypeToMlirType(
      shape.element_type(), context,
      layout.has_is_montgomery_form() && layout.is_montgomery_form());
  // TODO(chokobole): Take `major_to_minor` into account.
  auto dimensions_span = shape.dimensions();
  llvm::ArrayRef<int64_t> dimensions(dimensions_span.data(),
                                     dimensions_span.size());
  if (LayoutUtil::IsMonotonicWithDim0Major(layout)) {
    return mlir::MemRefType::get(dimensions, element_type);
  }

  llvm::SmallVector<int64_t> strides(dimensions.size(), 0);
  int64_t running = 1;
  for (int64_t d : layout.minor_to_major()) {
    strides[d] = running;
    // NOTE(chokobole): CHECK_GE(dimensions[d], 0) is used to allow for cases
    // where a dimension can legitimately have a size of zero. For example, an
    // empty array [2, 0, 3] is a valid shape.
    CHECK_GE(dimensions[d], 0);
    running *= dimensions[d];
  }

  auto strided = mlir::StridedLayoutAttr::get(context, /*offset=*/0, strides);
  return mlir::MemRefType::get(dimensions, element_type, strided);
}

namespace {

mlir::DenseIntElementsAttr CreateDenseIntElementsAttrFromVector(
    mlir::MLIRContext* context, const llvm::ArrayRef<int64_t> vector) {
  return mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get(vector.size(),
                                  mlir::IntegerType::get(context, 64)),
      vector);
}

}  // namespace

mlir::RankedTensorType ShapeToMlirTensorType(const Shape& shape,
                                             mlir::MLIRContext* context) {
  CHECK(shape.IsArray());
  const Layout& layout = shape.layout();
  mlir::Type element_type = PrimitiveTypeToMlirType(
      shape.element_type(), context,
      layout.has_is_montgomery_form() && layout.is_montgomery_form());

  auto dimensions_span = shape.dimensions();
  llvm::ArrayRef<int64_t> dimensions(dimensions_span.data(),
                                     dimensions_span.size());
  if (LayoutUtil::IsSparse(layout)) {
    auto convert_to_mlir_level = [](DimLevelType dlt, bool ordered,
                                    bool unique) {
      switch (dlt) {
        case DimLevelType::DIM_DENSE:
          return *mlir::sparse_tensor::buildLevelType(
              mlir::sparse_tensor::LevelFormat::Dense, ordered, unique);
        case DimLevelType::DIM_COMPRESSED:
          return *mlir::sparse_tensor::buildLevelType(
              mlir::sparse_tensor::LevelFormat::Compressed, ordered, unique);
        case DimLevelType::DIM_SINGLETON:
          return *mlir::sparse_tensor::buildLevelType(
              mlir::sparse_tensor::LevelFormat::Singleton, ordered, unique);
        case DimLevelType::DIM_LOOSE_COMPRESSED:
          return *mlir::sparse_tensor::buildLevelType(
              mlir::sparse_tensor::LevelFormat::LooseCompressed, ordered,
              unique);
        case DimLevelType_INT_MIN_SENTINEL_DO_NOT_USE_:
        case DimLevelType_INT_MAX_SENTINEL_DO_NOT_USE_:
          break;
      }
      ABSL_UNREACHABLE();
      return mlir::sparse_tensor::LevelType(0);
    };

    llvm::SmallVector<mlir::sparse_tensor::LevelType> lts;
    for (int i = 0; i < layout.dim_level_types_size(); ++i) {
      DimLevelType dlt = layout.dim_level_type(i);
      bool ordered =
          i < layout.dim_ordered_size() ? layout.dim_ordered(i) : true;
      bool unique = i < layout.dim_unique_size() ? layout.dim_unique(i) : true;
      lts.push_back(convert_to_mlir_level(dlt, ordered, unique));
    }
    absl::Span<const int64_t> ordering = layout.minor_to_major();
    llvm::SmallVector<int64_t> major_to_minor = {ordering.rbegin(),
                                                 ordering.rend()};
    mlir::AffineMap id_map =
        mlir::AffineMap::getPermutationMap(major_to_minor, context);
    auto encoding = mlir::sparse_tensor::SparseTensorEncodingAttr::get(
        context, lts, id_map, mlir::AffineMap(), 32, 32);
    return mlir::RankedTensorType::get(dimensions, element_type, encoding);
  } else {
    // Default layouts create a lot of clutter in the IR, so only add an
    // encoding when needed.
    mlir::Attribute layout = {};
    if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
      layout = CreateDenseIntElementsAttrFromVector(
          context, llvm::to_vector(shape.layout().minor_to_major()));
    }
    return mlir::RankedTensorType::get(dimensions, element_type, layout);
  }
}

std::vector<mlir::Type> ShapeToMlirTensorTypes(const Shape& shape,
                                               mlir::MLIRContext* context) {
  std::vector<mlir::Type> types;
  for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
    types.push_back(ShapeToMlirTensorType(shape.tuple_shapes(i), context));
  }
  return types;
}

}  // namespace zkx::mlir_utils
