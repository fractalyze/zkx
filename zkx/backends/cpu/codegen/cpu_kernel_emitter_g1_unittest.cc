#include "zkx/backends/cpu/codegen/group_test.h"

namespace zkx::cpu {

using GroupTypes = testing::Types<math::bn254::G1AffinePoint>;
TYPED_TEST_SUITE(GroupScalarUnaryTest, GroupTypes);

TYPED_TEST(GroupScalarUnaryTest, Convert) {
  this->SetUpConvert();
  this->RunAndVerify();
}

TYPED_TEST(GroupScalarUnaryTest, Negate) {
  this->SetUpNegate();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(GroupScalarBinaryTest, GroupTypes);

TYPED_TEST(GroupScalarBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST(GroupScalarBinaryTest, Double) {
  this->SetUpDouble();
  this->RunAndVerify();
}

TYPED_TEST(GroupScalarBinaryTest, Sub) {
  this->SetUpSub();
  this->RunAndVerify();
}

TYPED_TEST(GroupScalarBinaryTest, ScalarMul) {
  this->SetUpScalarMul();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(GroupR2TensorUnaryTest, GroupTypes);

TYPED_TEST(GroupR2TensorUnaryTest, Negate) {
  this->SetUpNegate();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(GroupR2TensorBinaryTest, GroupTypes);

TYPED_TEST(GroupR2TensorBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

}  // namespace zkx::cpu
