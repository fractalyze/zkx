#include "zkx/backends/cpu/codegen/field_test.h"

namespace zkx::cpu {

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

TYPED_TEST(FieldScalarUnaryTest, Inverse) {
  this->SetUpInverse();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldScalarBinaryTest, FieldTypes);

TYPED_TEST(FieldScalarBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Compare) {
  this->SetUpCompare();
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

TYPED_TEST(FieldScalarBinaryTest, PowWithSignedExponentShouldFail) {
  this->SetUpPowWithSignedExponentShouldFail();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarBinaryTest, Sub) {
  this->SetUpSub();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldR1TensorUnaryTest, FieldTypes);

TYPED_TEST(FieldR1TensorUnaryTest, BatchInverse) {
  this->SetUpBatchInverse();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorUnaryTest, FFT) {
  this->SetUpFFT();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorUnaryTest, FFTWithTwiddles) {
  this->SetUpFFTWithTwiddles();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorUnaryTest, IFFT) {
  this->SetUpIFFT();
  this->RunAndVerify();
}

TYPED_TEST(FieldR1TensorUnaryTest, IFFTWithTwiddles) {
  this->SetUpIFFTWithTwiddles();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldR2TensorBinaryTest, FieldTypes);

TYPED_TEST(FieldR2TensorBinaryTest, Add) { this->SetUpAdd(); }

TYPED_TEST_SUITE(FieldTest, FieldTypes);

TYPED_TEST(FieldTest, BroadcastScalar) {
  this->SetUpBroadcastScalar();
  this->RunAndVerify();
}

TYPED_TEST(FieldTest, BroadcastTensorR1ToR3WithD0) {
  this->SetUpBroadcastTensorR1ToR3WithD0();
  this->RunAndVerify();
}

TYPED_TEST(FieldTest, BroadcastTensorR1ToR3WithD1) {
  this->SetUpBroadcastTensorR1ToR3WithD1();
  this->RunAndVerify();
}

TYPED_TEST(FieldTest, BroadcastTensorR1ToR3WithD2) {
  this->SetUpBroadcastTensorR1ToR3WithD2();
  this->RunAndVerify();
}

TYPED_TEST(FieldTest, CSRMatrixVectorMultiplication) {
  this->SetUpCSRMatrixVectorMultiplication();
  this->RunAndVerify();
}

TYPED_TEST(FieldTest, Slice) {
  this->SetUpSlice();
  this->RunAndVerify();
}

}  // namespace zkx::cpu
