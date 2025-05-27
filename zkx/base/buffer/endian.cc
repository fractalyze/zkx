#include "zkx/base/buffer/endian.h"

namespace zkx::base {

std::string_view EndianToString(Endian endian) {
  switch (endian) {
    case Endian::kNative:
      return "Native";
    case Endian::kBig:
      return "Big";
    case Endian::kLittle:
      return "Little";
  }
  return "";
}

std::ostream& operator<<(std::ostream& os, Endian endian) {
  return os << EndianToString(endian);
}

}  // namespace zkx::base
