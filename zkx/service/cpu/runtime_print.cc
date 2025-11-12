#include "zkx/service/cpu/runtime_print.h"

#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "zkx/math/elliptic_curve/bn/bn254/fr.h"
#include "zkx/math/elliptic_curve/bn/bn254/g1.h"
#include "zkx/math/elliptic_curve/bn/bn254/g2.h"

namespace zkx::cpu {

template <typename T>
void PrintMemref(void* memref_in) {
  impl::printMemRef(*reinterpret_cast<UnrankedMemRefType<T>*>(memref_in));
}

template <int N, typename T>
void PrintMemref(void* memref_in) {
  if constexpr (std::is_same_v<T, zkx::math::bn254::G1AffinePoint> ||
                std::is_same_v<T, zkx::math::bn254::G1JacobianPoint> ||
                std::is_same_v<T, zkx::math::bn254::G1PointXyzz>) {
    impl::printMemRef(
        *reinterpret_cast<StridedMemRefType<zkx::math::bn254::Fq, N + 1>*>(
            memref_in));
  } else if constexpr (  // NOLINT(readability/braces)
      std::is_same_v<T, zkx::math::bn254::G2AffinePoint> ||
      std::is_same_v<T, zkx::math::bn254::G2JacobianPoint> ||
      std::is_same_v<T, zkx::math::bn254::G2PointXyzz>) {
    impl::printMemRef(
        *reinterpret_cast<StridedMemRefType<zkx::math::bn254::Fq, N + 2>*>(
            memref_in));
  } else {
    impl::printMemRef(*reinterpret_cast<StridedMemRefType<T, N>*>(memref_in));
  }
}

}  // namespace zkx::cpu

#define DEFINE_PRINT_MEMREF_FUNCTION(name, type)                        \
  void __zkx_cpu_runtime_PrintMemref##name(void* memref) {              \
    zkx::cpu::PrintMemref<type>(memref);                                \
  }                                                                     \
  void _mlir_ciface___zkx_cpu_runtime_PrintMemref##name(void* memref) { \
    zkx::cpu::PrintMemref<type>(memref);                                \
  }

DEFINE_PRINT_MEMREF_FUNCTION(Bn254Scalar, zkx::math::bn254::Fr)
DEFINE_PRINT_MEMREF_FUNCTION(Bn254G1Affine, zkx::math::bn254::G1AffinePoint)
DEFINE_PRINT_MEMREF_FUNCTION(Bn254G1Jacobian, zkx::math::bn254::G1JacobianPoint)
DEFINE_PRINT_MEMREF_FUNCTION(Bn254G1Xyzz, zkx::math::bn254::G1PointXyzz)
DEFINE_PRINT_MEMREF_FUNCTION(Bn254G2Affine, zkx::math::bn254::G2AffinePoint)
DEFINE_PRINT_MEMREF_FUNCTION(Bn254G2Jacobian, zkx::math::bn254::G2JacobianPoint)
DEFINE_PRINT_MEMREF_FUNCTION(Bn254G2Xyz, zkx::math::bn254::G2PointXyzz)

#undef DEFINE_PRINT_MEMREF_FUNCTION

#define DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(rank, name, type)      \
  void __zkx_cpu_runtime_PrintMemref##rank##D##name(void* memref) { \
    zkx::cpu::PrintMemref<rank, type>(memref);                      \
  }                                                                 \
  void _mlir_ciface___zkx_cpu_runtime_PrintMemref##rank##D##name(   \
      void* memref) {                                               \
    zkx::cpu::PrintMemref<rank, type>(memref);                      \
  }

DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254Scalar, zkx::math::bn254::Fr);
DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G1Affine,
                                     zkx::math::bn254::G1AffinePoint);
DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G1Jacobian,
                                     zkx::math::bn254::G1JacobianPoint);
DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G1Xyzz,
                                     zkx::math::bn254::G1PointXyzz);
DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G2Affine,
                                     zkx::math::bn254::G2AffinePoint);
DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G2Jacobian,
                                     zkx::math::bn254::G2JacobianPoint);
DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G2Xyz,
                                     zkx::math::bn254::G2PointXyzz);

#undef DEFINE_PRINT_STRIDED_MEMREF_FUNCTION
