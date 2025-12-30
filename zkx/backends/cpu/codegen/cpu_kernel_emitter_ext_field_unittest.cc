/* Copyright 2026 The ZKX Authors.

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

#include "zk_dtypes/include/all_types.h"

#include "zkx/backends/cpu/codegen/ext_field_test.h"
#include "zkx/backends/cpu/codegen/field_test.h"

namespace zkx::cpu {

using ExtensionFieldTypes = testing::Types<
    // clang-format off
    zk_dtypes::Mersenne312,
    zk_dtypes::Goldilocks3,
    zk_dtypes::Babybear4,
    zk_dtypes::Koalabear4
    // clang-format on
    >;

TYPED_TEST_SUITE(FieldScalarUnaryTest, ExtensionFieldTypes);

TYPED_TEST(FieldScalarUnaryTest, Negate) {
  this->SetUpNegate();
  this->RunAndVerify();
}

TYPED_TEST(FieldScalarUnaryTest, Inverse) {
  this->SetUpInverse();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(ExtFieldScalarBinaryTest, ExtensionFieldTypes);

TYPED_TEST(ExtFieldScalarBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

TYPED_TEST(ExtFieldScalarBinaryTest, CompareEq) {
  this->SetUpCompareEq();
  this->RunAndVerify();
}

TYPED_TEST(ExtFieldScalarBinaryTest, CompareNe) {
  this->SetUpCompareNe();
  this->RunAndVerify();
}

TYPED_TEST(ExtFieldScalarBinaryTest, Div) {
  this->SetUpDiv();
  this->RunAndVerify();
}

TYPED_TEST(ExtFieldScalarBinaryTest, Double) {
  this->SetUpDouble();
  this->RunAndVerify();
}

TYPED_TEST(ExtFieldScalarBinaryTest, Mul) {
  this->SetUpMul();
  this->RunAndVerify();
}

TYPED_TEST(ExtFieldScalarBinaryTest, Square) {
  this->SetUpSquare();
  this->RunAndVerify();
}

TYPED_TEST(ExtFieldScalarBinaryTest, Sub) {
  this->SetUpSub();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(ExtFieldScalarTernaryTest, ExtensionFieldTypes);

TYPED_TEST(ExtFieldScalarTernaryTest, SelectTrue) {
  this->SetUpSelectTrue();
  this->RunAndVerify();
}

TYPED_TEST(ExtFieldScalarTernaryTest, SelectFalse) {
  this->SetUpSelectFalse();
  this->RunAndVerify();
}

TYPED_TEST_SUITE(FieldR2TensorBinaryTest, ExtensionFieldTypes);

TYPED_TEST(FieldR2TensorBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

}  // namespace zkx::cpu
