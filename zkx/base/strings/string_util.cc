#include "zkx/base/strings/string_util.h"

namespace zkx::base {

std::string_view BoolToString(bool b) { return b ? "true" : "false"; }

}  // namespace zkx::base
