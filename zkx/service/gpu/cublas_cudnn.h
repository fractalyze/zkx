/* Copyright 2021 The OpenXLA Authors.
Copyright 2026 The ZKX Authors.

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

#ifndef ZKX_SERVICE_GPU_CUBLAS_CUDNN_H_
#define ZKX_SERVICE_GPU_CUBLAS_CUDNN_H_

#include <string>
#include <string_view>

#include "absl/status/statusor.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"

namespace zkx {
// TODO(zkx): This is a placeholder for HloCustomCallInstruction which has not
// been ported to zkx yet. When HloCustomCallInstruction is available, remove
// this forward declaration.
class HloCustomCallInstruction;
}  // namespace zkx

namespace zkx::gpu {

// Different types of convolutions supported by cudnn.
//
// A way to think about these is that a convolution is defined by three arrays
// -- the "input", the "filter", and the "output" -- and given any two of these,
// we can compute the third.
enum class CudnnConvKind {
  kForward,            // input  + filter => output
  kBackwardInput,      // filter + output => input
  kBackwardFilter,     // input  + output => filter
  kForwardActivation,  // activation(conv(input, filter) + broadcast(bias) +
                       // (optionally) side_input) => output
  kForwardGraph,       // pointwise(...pointwise(conv(input, filter))...)
                       // => output
};

absl::StatusOr<CudnnConvKind> GetCudnnConvKind(
    const HloCustomCallInstruction* instr);

// Converts a CudnnConvKind value to a string.
std::string CudnnConvKindToString(CudnnConvKind kind);

// Matrix multiplication rewritten into a GEMM custom call.
bool IsCublasGemm(const HloInstruction& hlo);

// Matrix multiplication that calls into legacy cublas.
bool IsLegacyCublasMatmul(const HloInstruction& hlo);

// Matrix multiplication that calls into cublasLt.
bool IsCublasLtMatmul(const HloInstruction& hlo);

// Scaled matrix multiplication in FP8. Calls into cublasLt.
bool IsCublasLtMatmulF8(const HloInstruction& hlo);

// A call to cuBLAS general matrix multiplication API.
extern const std::string_view kGemmCallTarget;

// A call to cuBLAS Lt API matrix multiplication.
extern const std::string_view kCublasLtMatmulCallTarget;

// A call to cuBLASLt for scaled matrix multiplication in FP8.
extern const std::string_view kCublasLtMatmulF8CallTarget;

// cuDNN convolution call targets.
extern const std::string_view kCudnnConvForwardCallTarget;
extern const std::string_view kCudnnConvBackwardInputCallTarget;
extern const std::string_view kCudnnConvBackwardFilterCallTarget;
extern const std::string_view kCudnnConvBiasActivationForwardCallTarget;
extern const std::string_view kCudnnConvForwardGraphCallTarget;

// Returns true if `hlo` will be implemented as a call to a cuDNN convolution
// routine.
bool IsCustomCallToDnnConvolution(const HloInstruction& hlo);

// CUB library calls.
extern const std::string_view kCubDeviceRadixSortTarget;

bool IsCubDeviceRadixSort(const HloInstruction& hlo);

// TopK custom call target.
extern const std::string_view kTopKCustomCallTarget;

bool IsCustomCallToTopK(const HloInstruction& hlo);

}  // namespace zkx::gpu

#endif  // ZKX_SERVICE_GPU_CUBLAS_CUDNN_H_
