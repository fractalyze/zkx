#include "zkx/backends/cpu/codegen/small_int_test.h"

namespace zkx::cpu {

using SmallIntTypes = testing::Types<u2, s2>;
TYPED_TEST_SUITE(SmallIntScalarBinaryTest, SmallIntTypes);

TYPED_TEST(SmallIntScalarBinaryTest, Add) {
  this->SetUpAdd();
  this->RunAndVerify();
}

}  // namespace zkx::cpu
