#include "zkx/backends/cpu/codegen/msm_test.h"

namespace zkx::cpu {

using GroupTypes = testing::Types<math::bn254::G1AffinePoint>;
TYPED_TEST_SUITE(MSMTest, GroupTypes);

TYPED_TEST(MSMTest, G1MSM) {
  this->SetUpMSM();
  this->RunAndVerify();
}

}  // namespace zkx::cpu
