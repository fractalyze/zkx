#ifndef ZKX_BASE_BUFFER_ENDIAN_H_
#define ZKX_BASE_BUFFER_ENDIAN_H_

#include <ostream>
#include <string_view>

namespace zkx::base {

enum class Endian {
  kNative,
  kBig,
  kLittle,
};

std::string_view EndianToString(Endian endian);

std::ostream& operator<<(std::ostream& os, Endian endian);

}  // namespace zkx::base

#endif  // ZKX_BASE_BUFFER_ENDIAN_H_
