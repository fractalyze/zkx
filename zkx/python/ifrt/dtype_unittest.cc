// Copyright 2024 The OpenXLA Authors.
// Copyright 2025 The ZKX Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "zkx/python/ifrt/dtype.h"

#include <optional>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "xla/tsl/platform/statusor.h"
#include "zkx/python/ifrt/serdes_test_util.h"

namespace zkx::ifrt {
namespace {

class DTypeSerDesTest : public testing::TestWithParam<SerDesVersion> {
 public:
  DTypeSerDesTest() : version_(GetParam()) {}

  SerDesVersion version() const { return version_; }

 private:
  SerDesVersion version_;
};

TEST_P(DTypeSerDesTest, FromToFromProto) {
  for (int i = 0; i < DTypeProto::Kind_descriptor()->value_count(); ++i) {
    DTypeProto proto;
    proto.set_version_number(version().version_number().value());
    proto.set_kind(static_cast<DTypeProto::Kind>(
        DTypeProto::Kind_descriptor()->value(i)->number()));
    TF_ASSERT_OK_AND_ASSIGN(DType dtype, DType::FromProto(proto));
    TF_ASSERT_OK_AND_ASSIGN(DType dtype_copy,
                            DType::FromProto(dtype.ToProto()));
    EXPECT_EQ(dtype_copy, dtype);
  }
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, DTypeSerDesTest,
    testing::ValuesIn(test_util::AllSupportedSerDesVersions()));

TEST(DTypeTest, ByteSize) {
  for (const auto& [kind, byte_size] :
       std::vector<std::tuple<DType::Kind, int>>({
           {DType::kS2, -1},
           {DType::kU2, -1},
           {DType::kS4, -1},
           {DType::kU4, -1},
           {DType::kPred, 1},
           {DType::kS8, 1},
           {DType::kU8, 1},
           {DType::kS16, 2},
           {DType::kU16, 2},
           {DType::kS32, 4},
           {DType::kU32, 4},
           {DType::kS64, 8},
           {DType::kU64, 8},
           {DType::kKoalabear, 4},
           {DType::kKoalabearStd, 4},
           {DType::kBabybear, 4},
           {DType::kBabybearStd, 4},
           {DType::kMersenne31, 4},
           {DType::kMersenne31Std, 4},
           {DType::kGoldilocks, 8},
           {DType::kGoldilocksStd, 8},
           {DType::kBn254Sf, 32},
           {DType::kBn254SfStd, 32},
           {DType::kBn254G1Affine, 64},
           {DType::kBn254G1AffineStd, 64},
           {DType::kBn254G1Jacobian, 128},
           {DType::kBn254G1JacobianStd, 128},
           {DType::kBn254G1Xyzz, 256},
           {DType::kBn254G1XyzzStd, 256},
           {DType::kBn254G2Affine, 128},
           {DType::kBn254G2AffineStd, 128},
           {DType::kBn254G2Jacobian, 192},
           {DType::kBn254G2JacobianStd, 192},
           {DType::kBn254G2Xyzz, 256},
           {DType::kBn254G2XyzzStd, 256},
           {DType::kToken, -1},
           {DType::kInvalid, -1},
           {DType::kString, -1},
       })) {
    EXPECT_EQ(DType(kind).byte_size(),
              byte_size == -1 ? std::nullopt : std::make_optional(byte_size));
  }
}

TEST(DTypeTest, BitSize) {
  for (const auto& [kind, bit_size] :
       std::vector<std::tuple<DType::Kind, int>>({
           {DType::kS2, 2},
           {DType::kU2, 2},
           {DType::kS4, 4},
           {DType::kU4, 4},
           {DType::kPred, 8},
           {DType::kS8, 8},
           {DType::kU8, 8},
           {DType::kS16, 16},
           {DType::kU16, 16},
           {DType::kS32, 32},
           {DType::kU32, 32},
           {DType::kS64, 64},
           {DType::kU64, 64},
           {DType::kKoalabear, 32},
           {DType::kKoalabearStd, 32},
           {DType::kBabybear, 32},
           {DType::kBabybearStd, 32},
           {DType::kMersenne31, 32},
           {DType::kMersenne31Std, 32},
           {DType::kGoldilocks, 64},
           {DType::kGoldilocksStd, 64},
           {DType::kBn254Sf, 256},
           {DType::kBn254SfStd, 256},
           {DType::kBn254G1Affine, 512},
           {DType::kBn254G1AffineStd, 512},
           {DType::kBn254G1Jacobian, 768},
           {DType::kBn254G1JacobianStd, 768},
           {DType::kBn254G1Xyzz, 1024},
           {DType::kBn254G1XyzzStd, 1024},
           {DType::kBn254G2Affine, 1024},
           {DType::kBn254G2AffineStd, 1024},
           {DType::kBn254G2Jacobian, 1536},
           {DType::kBn254G2JacobianStd, 1536},
           {DType::kBn254G2Xyzz, 2048},
           {DType::kBn254G2XyzzStd, 2048},
           {DType::kToken, -1},
           {DType::kInvalid, -1},
           {DType::kString, -1},
       })) {
    EXPECT_EQ(DType(kind).bit_size(),
              bit_size == -1 ? std::nullopt : std::make_optional(bit_size));
  }
}

}  // namespace
}  // namespace zkx::ifrt
