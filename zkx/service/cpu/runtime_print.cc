/* Copyright 2025 The ZKX Authors.

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

#include "zkx/service/cpu/runtime_print.h"

#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "zkx/math/elliptic_curve/bn/bn254/fr.h"
#include "zkx/math/elliptic_curve/bn/bn254/g1.h"
#include "zkx/math/elliptic_curve/bn/bn254/g2.h"
#include "zkx/math/field/babybear/babybear.h"
#include "zkx/math/field/goldilocks/goldilocks.h"
#include "zkx/math/field/koalabear/koalabear.h"
#include "zkx/math/field/mersenne31/mersenne31.h"

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

DEFINE_PRINT_MEMREF_FUNCTION(Koalabear, zkx::math::Koalabear)
DEFINE_PRINT_MEMREF_FUNCTION(Babybear, zkx::math::Babybear)
DEFINE_PRINT_MEMREF_FUNCTION(Mersenne31, zkx::math::Mersenne31)
DEFINE_PRINT_MEMREF_FUNCTION(Goldilocks, zkx::math::Goldilocks)
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

DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(1, Koalabear, zkx::math::Koalabear);
DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(1, Babybear, zkx::math::Babybear);
DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(1, Mersenne31, zkx::math::Mersenne31);
DEFINE_PRINT_STRIDED_MEMREF_FUNCTION(1, Goldilocks, zkx::math::Goldilocks);
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
