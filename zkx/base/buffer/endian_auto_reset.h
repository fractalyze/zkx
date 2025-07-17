#ifndef ZKX_BASE_BUFFER_ENDIAN_AUTO_RESET_H_
#define ZKX_BASE_BUFFER_ENDIAN_AUTO_RESET_H_

#include "zkx/base/buffer/read_only_buffer.h"

namespace zkx::base {

struct EndianAutoReset {
  explicit EndianAutoReset(const base::ReadOnlyBuffer& buffer,
                           base::Endian endian)
      : buffer(buffer), old_endian(buffer.endian()) {
    buffer.set_endian(endian);
  }
  ~EndianAutoReset() { buffer.set_endian(old_endian); }

  const base::ReadOnlyBuffer& buffer;
  base::Endian old_endian;
};

}  // namespace zkx::base

#endif  // ZKX_BASE_BUFFER_ENDIAN_AUTO_RESET_H_
