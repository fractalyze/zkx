#include <stdint.h>

#include "zkx/backends/cpu/codegen/int_test.h"

namespace zkx::cpu {

using IntTypes = testing::Types<int32_t, uint32_t>;
TYPED_TEST_SUITE(IntScalarUnaryTest, IntTypes);

TYPED_TEST(IntScalarUnaryTest, ConvertUp) {
  this->SetUpConvertUp();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarUnaryTest, ConvertDown) {
  this->SetUpConvertDown();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarUnaryTest, Negate) {
  if (std::is_signed_v<TypeParam>) {
    this->SetUpNegate();
    this->RunAndVerify();
  } else {
    GTEST_SKIP() << "Skipping test for unsigned type";
  }
}

TYPED_TEST_SUITE(IntScalarBinaryTest, IntTypes);

TYPED_TEST(IntScalarBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Sub) {
  this->SetUpSub();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Mul) {
  this->SetUpMul();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Div) {
  this->SetUpDiv();
  this->RunAndVerify();
}

// TODO(chokobole): Add tests for power.

TYPED_TEST_SUITE(IntR2TensorBinaryTest, IntTypes);

TYPED_TEST(IntR2TensorBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(IntTest, IntTypes);

TYPED_TEST(IntTest, Slice) {
  this->SetUpSlice();
  this->RunAndVerify();
}

}  // namespace zkx::cpu
