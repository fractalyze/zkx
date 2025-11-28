/* Copyright 2017 The OpenXLA Authors.
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

#include "zkx/primitive_util.h"

#include <stddef.h>

#include "absl/container/flat_hash_map.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/strings/ascii.h"

namespace zkx::primitive_util {

PrimitiveType UnsignedIntegralTypeForBitWidth(int64_t src_bitwidth) {
  switch (src_bitwidth) {
    case 1:
      return U1;
    case 2:
      return U2;
    case 4:
      return U4;
    case 8:
      return U8;
    case 16:
      return U16;
    case 32:
      return U32;
    case 64:
      return U64;
    default:
      return PRIMITIVE_TYPE_INVALID;
  }
}

PrimitiveType SignedIntegralTypeForBitWidth(int64_t src_bitwidth) {
  switch (src_bitwidth) {
    // NOTE(chokobole): Original XLA code doesn't have S1 branch. See
    // https://github.com/openxla/xla/blob/3191140/xla/primitive_util.cc#L122-L139
    case 1:
      return S1;
    case 2:
      return S2;
    case 4:
      return S4;
    case 8:
      return S8;
    case 16:
      return S16;
    case 32:
      return S32;
    case 64:
      return S64;
    default:
      return PRIMITIVE_TYPE_INVALID;
  }
}

namespace {

// Class to memoize the computation of
//   absl::AsciiStrToLower(PrimitiveType_Name(p))
// for all PrimitiveType values "p"
//
// zkx::OPAQUE_TYPE canonically maps to the string "opaque" -- the only reason
// it's called OPAQUE_TYPE is to avoid clashing with a windows.h macro.
class PrimitiveTypeNameGenerator {
 public:
  PrimitiveTypeNameGenerator() {
    for (size_t idx = 0; idx < std::size(lowercase_name_); ++idx) {
      PrimitiveType t = static_cast<PrimitiveType>(idx + PrimitiveType_MIN);
      if (t == OPAQUE_TYPE) {
        lowercase_name_[idx] = "opaque";
      } else if (t == KOALABEAR) {
        lowercase_name_[idx] = "koalabear";
      } else if (t == KOALABEAR_STD) {
        lowercase_name_[idx] = "koalabear_std";
      } else if (t == BABYBEAR) {
        lowercase_name_[idx] = "babybear";
      } else if (t == BABYBEAR_STD) {
        lowercase_name_[idx] = "babybear_std";
      } else if (t == MERSENNE31) {
        lowercase_name_[idx] = "mersenne31";
      } else if (t == MERSENNE31_STD) {
        lowercase_name_[idx] = "mersenne31_std";
      } else if (t == GOLDILOCKS) {
        lowercase_name_[idx] = "goldilocks";
      } else if (t == GOLDILOCKS_STD) {
        lowercase_name_[idx] = "goldilocks_std";
      } else if (t == BN254_SF) {
        lowercase_name_[idx] = "bn254.sf";
      } else if (t == BN254_SF_STD) {
        lowercase_name_[idx] = "bn254.sf_std";
      } else if (t == BN254_G1_AFFINE) {
        lowercase_name_[idx] = "bn254.g1_affine";
      } else if (t == BN254_G1_AFFINE_STD) {
        lowercase_name_[idx] = "bn254.g1_affine_std";
      } else if (t == BN254_G1_JACOBIAN) {
        lowercase_name_[idx] = "bn254.g1_jacobian";
      } else if (t == BN254_G1_JACOBIAN_STD) {
        lowercase_name_[idx] = "bn254.g1_jacobian_std";
      } else if (t == BN254_G1_XYZZ) {
        lowercase_name_[idx] = "bn254.g1_xyzz";
      } else if (t == BN254_G1_XYZZ_STD) {
        lowercase_name_[idx] = "bn254.g1_xyzz_std";
      } else if (t == BN254_G2_AFFINE) {
        lowercase_name_[idx] = "bn254.g2_affine";
      } else if (t == BN254_G2_AFFINE_STD) {
        lowercase_name_[idx] = "bn254.g2_affine_std";
      } else if (t == BN254_G2_JACOBIAN) {
        lowercase_name_[idx] = "bn254.g2_jacobian";
      } else if (t == BN254_G2_JACOBIAN_STD) {
        lowercase_name_[idx] = "bn254.g2_jacobian_std";
      } else if (t == BN254_G2_XYZZ) {
        lowercase_name_[idx] = "bn254.g2_xyzz";
      } else if (t == BN254_G2_XYZZ_STD) {
        lowercase_name_[idx] = "bn254.g2_xyzz_std";
      } else if (PrimitiveType_IsValid(t)) {
        lowercase_name_[idx] = absl::AsciiStrToLower(PrimitiveType_Name(t));
      }
    }
  }
  std::string_view LowercaseName(PrimitiveType t) {
    CHECK_GE(t, PrimitiveType_MIN);
    CHECK_LE(t, PrimitiveType_MAX);
    CHECK(PrimitiveType_IsValid(t))
        << "Invalid PrimitiveType: " << static_cast<int>(t);
    return lowercase_name_[t - PrimitiveType_MIN];
  }

 private:
  std::string lowercase_name_[PrimitiveType_MAX - PrimitiveType_MIN + 1];
};

}  // namespace

std::string_view LowercasePrimitiveTypeName(PrimitiveType s) {
  static auto* gen = absl::IgnoreLeak(new PrimitiveTypeNameGenerator());
  return gen->LowercaseName(s);
}

namespace {

// Returns a map from lower-case primitive type name to primitive type.
//
// Due to Postel's Law considerations, both "opaque" and "opaque_type" map to
// the zkx::OPAQUE_TYPE enumerator.
const absl::flat_hash_map<std::string, PrimitiveType>&
GetPrimitiveTypeStringMap() {
  static absl::flat_hash_map<std::string, PrimitiveType>* name_to_type = [] {
    static auto* map =
        absl::IgnoreLeak(new absl::flat_hash_map<std::string, PrimitiveType>);
    for (int i = 0; i < PrimitiveType_ARRAYSIZE; i++) {
      if (PrimitiveType_IsValid(i) && i != PRIMITIVE_TYPE_INVALID) {
        auto value = static_cast<PrimitiveType>(i);
        (*map)[LowercasePrimitiveTypeName(value)] = value;
      }
    }
    (*map)["opaque"] = OPAQUE_TYPE;
    return map;
  }();
  return *name_to_type;
}

}  // namespace

absl::StatusOr<PrimitiveType> StringToPrimitiveType(std::string_view name) {
  const auto& map = GetPrimitiveTypeStringMap();
  auto found = map.find(name);
  if (found == map.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid element type string: \"%s\".", name));
  }
  return found->second;
}

bool IsPrimitiveTypeName(std::string_view name) {
  const auto& map = GetPrimitiveTypeStringMap();
  auto found = map.find(name);
  return found != map.end();
}

}  // namespace zkx::primitive_util
