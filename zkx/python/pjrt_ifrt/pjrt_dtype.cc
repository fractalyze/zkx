/* Copyright 2024 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/python/pjrt_ifrt/pjrt_dtype.h"

#include "absl/strings/str_format.h"

namespace zkx::ifrt {

absl::StatusOr<PrimitiveType> ToPrimitiveType(DType dtype) {
  switch (dtype.kind()) {
#define CASE(DT, PT)                                                       \
  case DT:                                                                 \
    static_assert(PT == static_cast<PrimitiveType>(static_cast<int>(DT))); \
    return PT
    CASE(DType::kInvalid, PrimitiveType::PRIMITIVE_TYPE_INVALID);
    CASE(DType::kPred, PrimitiveType::PRED);
    CASE(DType::kS2, PrimitiveType::S2);
    CASE(DType::kS4, PrimitiveType::S4);
    CASE(DType::kS8, PrimitiveType::S8);
    CASE(DType::kS16, PrimitiveType::S16);
    CASE(DType::kS32, PrimitiveType::S32);
    CASE(DType::kS64, PrimitiveType::S64);
    CASE(DType::kU2, PrimitiveType::U2);
    CASE(DType::kU4, PrimitiveType::U4);
    CASE(DType::kU8, PrimitiveType::U8);
    CASE(DType::kU16, PrimitiveType::U16);
    CASE(DType::kU32, PrimitiveType::U32);
    CASE(DType::kU64, PrimitiveType::U64);
    CASE(DType::kToken, PrimitiveType::TOKEN);
    CASE(DType::kOpaque, PrimitiveType::OPAQUE_TYPE);
    CASE(DType::kKoalabear, PrimitiveType::KOALABEAR);
    CASE(DType::kKoalabearStd, PrimitiveType::KOALABEAR_STD);
    CASE(DType::kBabybear, PrimitiveType::BABYBEAR);
    CASE(DType::kBabybearStd, PrimitiveType::BABYBEAR_STD);
    CASE(DType::kMersenne31, PrimitiveType::MERSENNE31);
    CASE(DType::kMersenne31Std, PrimitiveType::MERSENNE31_STD);
    CASE(DType::kGoldilocks, PrimitiveType::GOLDILOCKS);
    CASE(DType::kGoldilocksStd, PrimitiveType::GOLDILOCKS_STD);
    CASE(DType::kBn254Scalar, PrimitiveType::BN254_SCALAR);
    CASE(DType::kBn254ScalarStd, PrimitiveType::BN254_SCALAR_STD);
    CASE(DType::kBn254G1Affine, PrimitiveType::BN254_G1_AFFINE);
    CASE(DType::kBn254G1AffineStd, PrimitiveType::BN254_G1_AFFINE_STD);
    CASE(DType::kBn254G1Jacobian, PrimitiveType::BN254_G1_JACOBIAN);
    CASE(DType::kBn254G1JacobianStd, PrimitiveType::BN254_G1_JACOBIAN_STD);
    CASE(DType::kBn254G1Xyzz, PrimitiveType::BN254_G1_XYZZ);
    CASE(DType::kBn254G1XyzzStd, PrimitiveType::BN254_G1_XYZZ_STD);
    CASE(DType::kBn254G2Affine, PrimitiveType::BN254_G2_AFFINE);
    CASE(DType::kBn254G2AffineStd, PrimitiveType::BN254_G2_AFFINE_STD);
    CASE(DType::kBn254G2Jacobian, PrimitiveType::BN254_G2_JACOBIAN);
    CASE(DType::kBn254G2JacobianStd, PrimitiveType::BN254_G2_JACOBIAN_STD);
    CASE(DType::kBn254G2Xyzz, PrimitiveType::BN254_G2_XYZZ);
    CASE(DType::kBn254G2XyzzStd, PrimitiveType::BN254_G2_XYZZ_STD);
#undef CASE
    case DType::kString:
      return absl::InvalidArgumentError(
          absl::StrFormat("Not supported as ZKX PrimitiveType: %d",
                          static_cast<int>(dtype.kind())));
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Invalid DType: %d", static_cast<int>(dtype.kind())));
}

absl::StatusOr<DType> ToDType(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case PrimitiveType::PRIMITIVE_TYPE_INVALID:
    case PrimitiveType::PRED:
    case PrimitiveType::S2:
    case PrimitiveType::S4:
    case PrimitiveType::S8:
    case PrimitiveType::S16:
    case PrimitiveType::S32:
    case PrimitiveType::S64:
    case PrimitiveType::U2:
    case PrimitiveType::U4:
    case PrimitiveType::U8:
    case PrimitiveType::U16:
    case PrimitiveType::U32:
    case PrimitiveType::U64:
    case PrimitiveType::TOKEN:
    case PrimitiveType::OPAQUE_TYPE:
    case PrimitiveType::KOALABEAR:
    case PrimitiveType::KOALABEAR_STD:
    case PrimitiveType::BABYBEAR:
    case PrimitiveType::BABYBEAR_STD:
    case PrimitiveType::MERSENNE31:
    case PrimitiveType::MERSENNE31_STD:
    case PrimitiveType::GOLDILOCKS:
    case PrimitiveType::GOLDILOCKS_STD:
    case PrimitiveType::BN254_SCALAR:
    case PrimitiveType::BN254_SCALAR_STD:
    case PrimitiveType::BN254_G1_AFFINE:
    case PrimitiveType::BN254_G1_AFFINE_STD:
    case PrimitiveType::BN254_G1_JACOBIAN:
    case PrimitiveType::BN254_G1_JACOBIAN_STD:
    case PrimitiveType::BN254_G1_XYZZ:
    case PrimitiveType::BN254_G1_XYZZ_STD:
    case PrimitiveType::BN254_G2_AFFINE:
    case PrimitiveType::BN254_G2_AFFINE_STD:
    case PrimitiveType::BN254_G2_JACOBIAN:
    case PrimitiveType::BN254_G2_JACOBIAN_STD:
    case PrimitiveType::BN254_G2_XYZZ:
    case PrimitiveType::BN254_G2_XYZZ_STD:
      return DType(static_cast<DType::Kind>(static_cast<int>(primitive_type)));
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid ZKX PrimitiveType: %d", static_cast<int>(primitive_type)));
  }
}

}  // namespace zkx::ifrt
