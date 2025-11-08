#include <stdint.h>

#include "zkx/backends/cpu/codegen/int_test.h"

namespace zkx::cpu {

using IntTypes = testing::Types<int32_t, uint32_t>;
TYPED_TEST_SUITE(IntScalarUnaryTest, IntTypes);

TYPED_TEST(IntScalarUnaryTest, Abs) {
  if (std::is_signed_v<TypeParam>) {
    this->SetUpAbs();
    this->RunAndVerify();
  } else {
    GTEST_SKIP() << "Skipping test for unsigned type";
  }
}

TYPED_TEST(IntScalarUnaryTest, BitcastConvert) {
  this->SetUpBitcastConvert();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarUnaryTest, CountLeadingZeros) {
  this->SetUpCountLeadingZeros();
  this->RunAndVerify();
}

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

TYPED_TEST(IntScalarUnaryTest, Not) {
  this->SetUpNot();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarUnaryTest, PopulationCount) {
  this->SetUpPopulationCount();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarUnaryTest, Sign) {
  if (std::is_signed_v<TypeParam>) {
    this->SetUpSign();
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

TYPED_TEST(IntScalarBinaryTest, And) {
  this->SetUpAnd();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Compare) {
  this->SetUpCompare();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Div) {
  this->SetUpDiv();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Maximum) {
  this->SetUpMaximum();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Minimum) {
  this->SetUpMinimum();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Mul) {
  this->SetUpMul();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, ShiftLeft) {
  this->SetUpShiftLeft();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, ShiftRightArithmetic) {
  this->SetUpShiftRightArithmetic();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, ShiftRightLogical) {
  this->SetUpShiftRightLogical();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Sub) {
  this->SetUpSub();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Or) {
  this->SetUpOr();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Power) {
  this->SetUpPower();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Remainder) {
  this->SetUpRemainder();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarBinaryTest, Xor) {
  this->SetUpXor();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(IntScalarTernaryTest, IntTypes);

TYPED_TEST(IntScalarTernaryTest, Clamp) {
  this->SetUpClamp();
  this->RunAndVerify();
}

TYPED_TEST(IntScalarTernaryTest, Select) {
  this->SetUpSelect();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(IntR2TensorBinaryTest, IntTypes);

TYPED_TEST(IntR2TensorBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(IntTest, IntTypes);

TYPED_TEST(IntTest, BroadcastScalar) {
  this->SetUpBroadcastScalar();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, BroadcastTensorR1ToR3WithD0) {
  this->SetUpBroadcastTensorR1ToR3WithD0();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, BroadcastTensorR1ToR3WithD1) {
  this->SetUpBroadcastTensorR1ToR3WithD1();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, BroadcastTensorR1ToR3WithD2) {
  this->SetUpBroadcastTensorR1ToR3WithD2();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Concatenate) {
  this->SetUpConcatenate();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Conditional) {
  this->SetUpConditional();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, IotaWithD0) {
  this->SetUpIotaWithD0();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, IotaWithD1) {
  this->SetUpIotaWithD1();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Pad) {
  this->SetUpPad();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Reshape) {
  this->SetUpReshape();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Reverse) {
  this->SetUpReverse();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Slice) {
  this->SetUpSlice();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, Transpose) {
  this->SetUpTranspose();
  this->RunAndVerify();
}

TYPED_TEST(IntTest, While) {
  this->SetUpWhile();
  this->RunAndVerify();
}

}  // namespace zkx::cpu
