#include "zkx/base/strings/string_util.h"

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"

namespace zkx::base {
namespace {

constexpr const char* k0x = "0x";

}  // namespace

std::string_view BoolToString(bool b) { return b ? "true" : "false"; }

std::string ToHexStringWithLeadingZero(const std::string& str, size_t num) {
  CHECK_GE(num, str.size());
  std::string_view sv = str;
  if (absl::ConsumePrefix(&sv, k0x)) {
    return absl::StrCat(k0x, std::string(num - sv.size(), '0'), sv);
  } else {
    return absl::StrCat(std::string(num - sv.size(), '0'), sv);
  }
}

std::string MaybePrepend0x(std::string_view str) {
  if (absl::StartsWith(str, k0x)) {
    return std::string(str);
  }
  return absl::StrCat(k0x, str);
}

}  // namespace zkx::base
