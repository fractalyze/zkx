/* Copyright 2022 The OpenXLA Authors.
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

#include "zkx/python/ifrt/dtype.h"

#include "absl/strings/str_cat.h"

namespace zkx::ifrt {

std::optional<int> DType::byte_size() const {
  switch (kind_) {
    case kS2:
    case kU2:
    case kS4:
    case kU4:
      // Smaller than a byte.
      return std::nullopt;
    case kPred:
    case kS8:
    case kU8:
      return 1;
    case kS16:
    case kU16:
      return 2;
    case kS32:
    case kU32:
      return 4;
    case kS64:
    case kU64:
      return 8;
    case kKoalabear:
    case kKoalabearStd:
    case kBabybear:
    case kBabybearStd:
    case kMersenne31:
    case kMersenne31Std:
      return 4;
    case kGoldilocks:
    case kGoldilocksStd:
      return 8;
    case kBn254Sf:
    case kBn254SfStd:
      return 32;
    case kBn254G1Affine:
    case kBn254G1AffineStd:
      return 64;
    case kBn254G1Jacobian:
    case kBn254G1JacobianStd:
      return 128;
    case kBn254G1Xyzz:
    case kBn254G1XyzzStd:
      return 256;
    case kBn254G2Affine:
    case kBn254G2AffineStd:
      return 128;
    case kBn254G2Jacobian:
    case kBn254G2JacobianStd:
      return 192;
    case kBn254G2Xyzz:
    case kBn254G2XyzzStd:
      return 256;
    case kToken:
    case kOpaque:
    case kInvalid:
    case kString:
      return std::nullopt;
  }
}

std::optional<int> DType::bit_size() const {
  switch (kind_) {
    case kS2:
    case kU2:
      return 2;
    case kS4:
    case kU4:
      return 4;
    case kPred:
    case kS8:
    case kU8:
      return 8;
    case kS16:
    case kU16:
      return 16;
    case kS32:
    case kU32:
      return 32;
    case kS64:
    case kU64:
      return 64;
    case kKoalabear:
    case kKoalabearStd:
    case kBabybear:
    case kBabybearStd:
    case kMersenne31:
    case kMersenne31Std:
      return 32;
    case kGoldilocks:
    case kGoldilocksStd:
      return 64;
    case kBn254Sf:
    case kBn254SfStd:
      return 256;
    case kBn254G1Affine:
    case kBn254G1AffineStd:
      return 512;
    case kBn254G1Jacobian:
    case kBn254G1JacobianStd:
      return 768;
    case kBn254G1Xyzz:
    case kBn254G1XyzzStd:
      return 1024;
    case kBn254G2Affine:
    case kBn254G2AffineStd:
      return 1024;
    case kBn254G2Jacobian:
    case kBn254G2JacobianStd:
      return 1536;
    case kBn254G2Xyzz:
    case kBn254G2XyzzStd:
      return 2048;
    case kToken:
    case kOpaque:
    case kInvalid:
    case kString:
      return std::nullopt;
  }
}

// static
absl::StatusOr<DType> DType::FromProto(const DTypeProto& dtype_proto) {
  const SerDesVersionNumber version_number(dtype_proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for DType deserialization"));
  }

  switch (dtype_proto.kind()) {
    case DTypeProto::KIND_PRED:
      return DType(DType::Kind::kPred);
    case DTypeProto::KIND_TOKEN:
      return DType(DType::Kind::kToken);
    case DTypeProto::KIND_OPAQUE:
      return DType(DType::Kind::kOpaque);
#define CASE(X)              \
  case DTypeProto::KIND_##X: \
    return DType(DType::Kind::k##X);
      CASE(S4);
      CASE(S8);
      CASE(S16);
      CASE(S32);
      CASE(S64);
      CASE(U4);
      CASE(U8);
      CASE(U16);
      CASE(U32);
      CASE(U64);
#undef CASE

#define CASE(X, Y)           \
  case DTypeProto::KIND_##X: \
    return DType(DType::Kind::k##Y);
      CASE(KOALABEAR, Koalabear);
      CASE(KOALABEAR_STD, KoalabearStd);
      CASE(BABYBEAR, Babybear);
      CASE(BABYBEAR_STD, BabybearStd);
      CASE(MERSENNE31, Mersenne31);
      CASE(MERSENNE31_STD, Mersenne31Std);
      CASE(GOLDILOCKS, Goldilocks);
      CASE(GOLDILOCKS_STD, GoldilocksStd);
      CASE(BN254_SF, Bn254Sf);
      CASE(BN254_SF_STD, Bn254SfStd);
      CASE(BN254_G1_AFFINE, Bn254G1Affine);
      CASE(BN254_G1_AFFINE_STD, Bn254G1AffineStd);
      CASE(BN254_G1_JACOBIAN, Bn254G1Jacobian);
      CASE(BN254_G1_JACOBIAN_STD, Bn254G1JacobianStd);
      CASE(BN254_G1_XYZZ, Bn254G1Xyzz);
      CASE(BN254_G1_XYZZ_STD, Bn254G1XyzzStd);
      CASE(BN254_G2_AFFINE, Bn254G2Affine);
      CASE(BN254_G2_AFFINE_STD, Bn254G2AffineStd);
      CASE(BN254_G2_JACOBIAN, Bn254G2Jacobian);
      CASE(BN254_G2_JACOBIAN_STD, Bn254G2JacobianStd);
      CASE(BN254_G2_XYZZ, Bn254G2Xyzz);
      CASE(BN254_G2_XYZZ_STD, Bn254G2XyzzStd);
#undef CASE
    case DTypeProto::KIND_STRING:
      return DType(DType::Kind::kString);
    default:
      return DType(DType::Kind::kInvalid);
  }
}

DTypeProto DType::ToProto(SerDesVersion version) const {
  // TODO(b/423702568): Change the return type to `absl::StatusOr<...>` for
  // graceful error handling.
  CHECK_GE(version.version_number(), SerDesVersionNumber(0))
      << "Unsupported " << version.version_number()
      << " for DType serialization";

  DTypeProto dtype_proto;
  dtype_proto.set_version_number(SerDesVersionNumber(0).value());

  switch (kind()) {
    case DType::Kind::kPred:
      dtype_proto.set_kind(DTypeProto::KIND_PRED);
      break;
    case DType::Kind::kToken:
      dtype_proto.set_kind(DTypeProto::KIND_TOKEN);
      break;
    case DType::Kind::kOpaque:
      dtype_proto.set_kind(DTypeProto::KIND_OPAQUE);
      break;
#define CASE(X)                                 \
  case DType::Kind::k##X:                       \
    dtype_proto.set_kind(DTypeProto::KIND_##X); \
    break;
      CASE(S4);
      CASE(S8);
      CASE(S16);
      CASE(S32);
      CASE(S64);
      CASE(U4);
      CASE(U8);
      CASE(U16);
      CASE(U32);
      CASE(U64);
#undef CASE

#define CASE(X, Y)                              \
  case DType::Kind::k##X:                       \
    dtype_proto.set_kind(DTypeProto::KIND_##Y); \
    break;
      CASE(Koalabear, KOALABEAR);
      CASE(KoalabearStd, KOALABEAR_STD);
      CASE(Babybear, BABYBEAR);
      CASE(BabybearStd, BABYBEAR_STD);
      CASE(Mersenne31, MERSENNE31);
      CASE(Mersenne31Std, MERSENNE31_STD);
      CASE(Goldilocks, GOLDILOCKS);
      CASE(GoldilocksStd, GOLDILOCKS_STD);
      CASE(Bn254Sf, BN254_SF);
      CASE(Bn254SfStd, BN254_SF_STD);
      CASE(Bn254G1Affine, BN254_G1_AFFINE);
      CASE(Bn254G1AffineStd, BN254_G1_AFFINE_STD);
      CASE(Bn254G1Jacobian, BN254_G1_JACOBIAN);
      CASE(Bn254G1JacobianStd, BN254_G1_JACOBIAN_STD);
      CASE(Bn254G1Xyzz, BN254_G1_XYZZ);
      CASE(Bn254G1XyzzStd, BN254_G1_XYZZ_STD);
      CASE(Bn254G2Affine, BN254_G2_AFFINE);
      CASE(Bn254G2AffineStd, BN254_G2_AFFINE_STD);
      CASE(Bn254G2Jacobian, BN254_G2_JACOBIAN);
      CASE(Bn254G2JacobianStd, BN254_G2_JACOBIAN_STD);
      CASE(Bn254G2Xyzz, BN254_G2_XYZZ);
      CASE(Bn254G2XyzzStd, BN254_G2_XYZZ_STD);
#undef CASE
    case DType::Kind::kString:
      dtype_proto.set_kind(DTypeProto::KIND_STRING);
      break;
    default:
      dtype_proto.set_kind(DTypeProto::KIND_UNSPECIFIED);
      break;
  }
  return dtype_proto;
}

std::string DType::DebugString() const {
  switch (kind_) {
    case kInvalid:
      return "INVALID";
    case kPred:
      return "PRED";
    case kS2:
      return "S2";
    case kS4:
      return "S4";
    case kS8:
      return "S8";
    case kS16:
      return "S16";
    case kS32:
      return "S32";
    case kS64:
      return "S64";
    case kU2:
      return "U2";
    case kU4:
      return "U4";
    case kU8:
      return "U8";
    case kU16:
      return "U16";
    case kU32:
      return "U32";
    case kU64:
      return "U64";
    case kToken:
      return "TOKEN";
    case kOpaque:
      return "OPAQUE";
    case kKoalabear:
      return "KOALABEAR";
    case kKoalabearStd:
      return "KOALABEAR_STD";
    case kBabybear:
      return "BABYBEAR";
    case kBabybearStd:
      return "BABYBEAR_STD";
    case kMersenne31:
      return "MERSENNE31";
    case kMersenne31Std:
      return "MERSENNE31_STD";
    case kGoldilocks:
      return "GOLDILOCKS";
    case kGoldilocksStd:
      return "GOLDILOCKS_STD";
    case kBn254Sf:
      return "BN254_SF";
    case kBn254SfStd:
      return "BN254_SF_STD";
    case kBn254G1Affine:
      return "BN254_G1_AFFINE";
    case kBn254G1AffineStd:
      return "BN254_G1_AFFINE_STD";
    case kBn254G1Jacobian:
      return "BN254_G1_JACOBIAN";
    case kBn254G1JacobianStd:
      return "BN254_G1_JACOBIAN_STD";
    case kBn254G1Xyzz:
      return "BN254_G1_XYZZ";
    case kBn254G1XyzzStd:
      return "BN254_G1_XYZZ_STD";
    case kBn254G2Affine:
      return "BN254_G2_AFFINE";
    case kBn254G2AffineStd:
      return "BN254_G2_AFFINE_STD";
    case kBn254G2Jacobian:
      return "BN254_G2_JACOBIAN";
    case kBn254G2JacobianStd:
      return "BN254_G2_JACOBIAN_STD";
    case kBn254G2Xyzz:
      return "BN254_G2_XYZZ";
    case kBn254G2XyzzStd:
      return "BN254_G2_XYZZ_STD";
    case kString:
      return "STRING";
    default:
      return absl::StrCat("UNKNOWN(", static_cast<int>(kind_), ")");
  }
}

std::ostream& operator<<(std::ostream& os, const DType& dtype) {
  return os << dtype.DebugString();
}

}  // namespace zkx::ifrt
