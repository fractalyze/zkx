#ifndef ZKX_BASE_STRINGS_STRING_UTIL_H_
#define ZKX_BASE_STRINGS_STRING_UTIL_H_

#include <string>

namespace zkx::base {

std::string_view BoolToString(bool b);

// Return a string that left padded with zero so that size of a hex number
// is a given `num`. For example,
//
//   - `str` = "0x1234" and `num` = 6 => "0x001234"
//   - `str` = "1234" and `num` = 6 => "001234"
std::string ToHexStringWithLeadingZero(std::string_view str, size_t num);

std::string MaybePrepend0x(std::string_view str);

}  // namespace zkx::base

#endif  // ZKX_BASE_STRINGS_STRING_UTIL_H_
