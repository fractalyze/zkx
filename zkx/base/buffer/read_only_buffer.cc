#include "zkx/base/buffer/read_only_buffer.h"

#include <string.h>

#include "absl/base/internal/endian.h"

namespace zkx::base {

absl::Status ReadOnlyBuffer::ReadAt(size_t buffer_offset, uint8_t* ptr,
                                    size_t size) const {
  // TODO(chokobole): add overflow check
  if (buffer_offset + size > buffer_len_) {
    return absl::OutOfRangeError(
        absl::Substitute("Read out of bounds: buffer_offset ($0) + size ($1) "
                         "exceeds buffer_len_ ($2)",
                         buffer_offset, size, buffer_len_));
  }
  const char* buffer = reinterpret_cast<const char*>(buffer_);
  memcpy(ptr, &buffer[buffer_offset], size);
  buffer_offset_ = buffer_offset + size;
  return absl::OkStatus();
}

#define READ_BE_AT(bytes, bits, type)                                  \
  absl::Status ReadOnlyBuffer::Read##bits##BEAt(size_t buffer_offset,  \
                                                type* ptr) const {     \
    /* TODO(chokobole): add overflow check */                          \
    if (buffer_offset + bytes > buffer_len_) {                         \
      return absl::OutOfRangeError(absl::Substitute(                   \
          "Read out of bounds: buffer_offset ($0) + bytes ($1) "       \
          "exceeds buffer_len_ ($2)",                                  \
          buffer_offset, bytes, buffer_len_));                         \
    }                                                                  \
    const char* buffer = reinterpret_cast<char*>(buffer_);             \
    type value = absl::big_endian::Load##bits(&buffer[buffer_offset]); \
    memcpy(ptr, &value, bytes);                                        \
    buffer_offset_ = buffer_offset + bytes;                            \
    return absl::OkStatus();                                           \
  }

READ_BE_AT(2, 16, uint16_t)
READ_BE_AT(4, 32, uint32_t)
READ_BE_AT(8, 64, uint64_t)

#undef READ_BE_AT

#define READ_LE_AT(bytes, bits, type)                                     \
  absl::Status ReadOnlyBuffer::Read##bits##LEAt(size_t buffer_offset,     \
                                                type* ptr) const {        \
    /* TODO(chokobole): add overflow check */                             \
    if (buffer_offset + bytes > buffer_len_) {                            \
      return absl::OutOfRangeError(absl::Substitute(                      \
          "Read out of bounds: buffer_offset ($0) + bytes ($1) "          \
          "exceeds buffer_len_ ($2)",                                     \
          buffer_offset, bytes, buffer_len_));                            \
    }                                                                     \
    const char* buffer = reinterpret_cast<const char*>(buffer_);          \
    type value = absl::little_endian::Load##bits(&buffer[buffer_offset]); \
    memcpy(ptr, &value, bytes);                                           \
    buffer_offset_ = buffer_offset + bytes;                               \
    return absl::OkStatus();                                              \
  }

READ_LE_AT(2, 16, uint16_t)
READ_LE_AT(4, 32, uint32_t)
READ_LE_AT(8, 64, uint64_t)

#undef READ_LE_AT

}  // namespace zkx::base
