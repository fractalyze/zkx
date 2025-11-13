#include "zkx/backends/gpu/codegen/field_test.h"

namespace zkx::gpu {

using FieldTypes = testing::Types<math::bn254::Fr>;
TYPED_TEST_SUITE(FieldScalarUnaryTest, FieldTypes);

TYPED_TEST(FieldScalarUnaryTest, Convert) {
  this->SetUpConvert();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarUnaryTest, Negate) {
  this->SetUpNegate();
  this->RunAndVerify();
}

// TODO(chokobole): Add FieldScalarInverse test after creating inverse operation
// in mhlo dialect.

TYPED_TEST_SUITE(FieldScalarBinaryTest, FieldTypes);

TYPED_TEST(FieldScalarBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Div) {
  this->SetUpDiv();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Mul) {
  this->SetUpMul();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Pow) {
  this->SetUpPow();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Sub) {
  this->SetUpSub();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldR2TensorBinaryTest, FieldTypes);

TYPED_TEST(FieldR2TensorBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

// TODO(chokobole): Add FFT, IFFT test
// TODO(chokobole): Add CSRMatrixVectorMultiplication test

}  // namespace zkx::gpu
