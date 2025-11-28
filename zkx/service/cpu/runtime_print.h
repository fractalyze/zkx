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

#ifndef ZKX_SERVICE_CPU_RUNTIME_PRINT_H_
#define ZKX_SERVICE_CPU_RUNTIME_PRINT_H_

extern "C" {

#define DECLARE_PRINT_MEMREF_FUNCTION(name)                      \
  extern void __zkx_cpu_runtime_PrintMemref##name(void* memref); \
  extern void _mlir_ciface___zkx_cpu_runtime_PrintMemref##name(void* memref)

DECLARE_PRINT_MEMREF_FUNCTION(Koalabear);
DECLARE_PRINT_MEMREF_FUNCTION(Babybear);
DECLARE_PRINT_MEMREF_FUNCTION(Mersenne31);
DECLARE_PRINT_MEMREF_FUNCTION(Goldilocks);
DECLARE_PRINT_MEMREF_FUNCTION(Bn254Sf);
DECLARE_PRINT_MEMREF_FUNCTION(Bn254G1Affine);
DECLARE_PRINT_MEMREF_FUNCTION(Bn254G1Jacobian);
DECLARE_PRINT_MEMREF_FUNCTION(Bn254G1Xyzz);
DECLARE_PRINT_MEMREF_FUNCTION(Bn254G2Affine);
DECLARE_PRINT_MEMREF_FUNCTION(Bn254G2Jacobian);
DECLARE_PRINT_MEMREF_FUNCTION(Bn254G2Xyz);

#undef DECLARE_PRINT_MEMREF_FUNCTION

#define DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(rank, name)                 \
  extern void __zkx_cpu_runtime_PrintMemref##rank##D##name(void* memref); \
  extern void _mlir_ciface___zkx_cpu_runtime_PrintMemref##rank##D##name(  \
      void* memref)

DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(1, Koalabear);
DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(1, Babybear);
DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(1, Mersenne31);
DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(1, Goldilocks);
DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254Sf);
DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G1Affine);
DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G1Jacobian);
DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G1Xyzz);
DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G2Affine);
DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G2Jacobian);
DECLARE_PRINT_STRIDED_MEMREF_FUNCTION(1, Bn254G2Xyz);

#undef DECLARE_PRINT_STRIDED_MEMREF_FUNCTION
}

#endif  // ZKX_SERVICE_CPU_RUNTIME_PRINT_H_
