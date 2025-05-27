#include "zkx/base/random.h"

namespace zkx::base {

absl::BitGen& GetAbslBitGen() {
  static absl::BitGen bitgen;
  return bitgen;
}

}  // namespace zkx::base
