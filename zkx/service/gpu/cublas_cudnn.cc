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

#include "zkx/service/gpu/cublas_cudnn.h"

#include <string>

#include "absl/status/status.h"

#include "zkx/hlo/ir/hlo_instruction.h"
#include "zkx/hlo/ir/hlo_instructions.h"
#include "zkx/hlo/ir/hlo_opcode.h"
#include "zkx/util.h"

namespace zkx::gpu {

// TODO(zkx): These functions require HloCustomCallInstruction to be ported.
// Currently stubbed to return false since custom_call_target() is not
// available.

bool IsCublasGemm(const HloInstruction& hlo) {
  return IsLegacyCublasMatmul(hlo) || IsCublasLtMatmul(hlo) ||
         IsCublasLtMatmulF8(hlo);
}

bool IsLegacyCublasMatmul(const HloInstruction& hlo) {
  // TODO(zkx): Implement when HloCustomCallInstruction is available.
  (void)hlo;
  return false;
}

bool IsCublasLtMatmul(const HloInstruction& hlo) {
  // TODO(zkx): Implement when HloCustomCallInstruction is available.
  (void)hlo;
  return false;
}

bool IsCublasLtMatmulF8(const HloInstruction& hlo) {
  // TODO(zkx): Implement when HloCustomCallInstruction is available.
  (void)hlo;
  return false;
}

const std::string_view kGemmCallTarget = "__cublas$gemm";
const std::string_view kCublasLtMatmulCallTarget = "__cublas$lt$matmul";
const std::string_view kCublasLtMatmulF8CallTarget = "__cublas$lt$matmul$f8";

const std::string_view kCudnnConvForwardCallTarget = "__cudnn$convForward";
const std::string_view kCudnnConvBackwardInputCallTarget =
    "__cudnn$convBackwardInput";
const std::string_view kCudnnConvBackwardFilterCallTarget =
    "__cudnn$convBackwardFilter";
const std::string_view kCudnnConvBiasActivationForwardCallTarget =
    "__cudnn$convBiasActivationForward";
const std::string_view kCudnnConvForwardGraphCallTarget =
    "__cudnn$convForwardGraph";

const std::string_view kCubDeviceRadixSortTarget = "__cub$DeviceRadixSort";

const std::string_view kTopKCustomCallTarget = "__gpu$TopK";

bool IsCustomCallToDnnConvolution(const HloInstruction& hlo) {
  // TODO(zkx): Implement when HloCustomCallInstruction is available.
  (void)hlo;
  return false;
}

bool IsCubDeviceRadixSort(const HloInstruction& hlo) {
  // TODO(zkx): Implement when HloCustomCallInstruction is available.
  (void)hlo;
  return false;
}

bool IsCustomCallToTopK(const HloInstruction& hlo) {
  // TODO(zkx): Implement when HloCustomCallInstruction is available.
  (void)hlo;
  return false;
}

absl::StatusOr<CudnnConvKind> GetCudnnConvKind(
    const HloCustomCallInstruction* instr) {
  // TODO(zkx): Implement when HloCustomCallInstruction is available.
  (void)instr;
  return absl::InternalError(
      "HloCustomCallInstruction not yet implemented in zkx");
}

std::string CudnnConvKindToString(CudnnConvKind kind) {
  switch (kind) {
    case CudnnConvKind::kForward:
      return "forward";
    case CudnnConvKind::kBackwardFilter:
      return "backward_filter";
    case CudnnConvKind::kBackwardInput:
      return "backward_input";
    case CudnnConvKind::kForwardActivation:
      return "forward with activation";
    case CudnnConvKind::kForwardGraph:
      return "forward with pointwise operations";
  }
}

}  // namespace zkx::gpu
