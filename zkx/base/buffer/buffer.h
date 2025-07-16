#ifndef ZKX_BASE_BUFFER_BUFFER_H_
#define ZKX_BASE_BUFFER_BUFFER_H_

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/base/buffer/read_only_buffer.h"

namespace zkx::base {

class Buffer : public ReadOnlyBuffer {
 public:
  Buffer() = default;
  Buffer(void* buffer, size_t buffer_len)
      : ReadOnlyBuffer(buffer, buffer_len) {}
  Buffer(const Buffer& other) = delete;
  Buffer& operator=(const Buffer& other) = delete;
  Buffer(Buffer&& other) = default;
  Buffer& operator=(Buffer&& other) = default;
  virtual ~Buffer() = default;

  // NOTE(chokobole): Due to the existence of a constant getter named `buffer()`
  // in the parent class, the name was chosen in snake case.
  using ReadOnlyBuffer::buffer;
  void* buffer() { return buffer_; }

  absl::Status Write(const uint8_t* ptr, size_t size) {
    return WriteAt(buffer_offset_, ptr, size);
  }

  template <typename T>
  absl::Status Write(const T& value) {
    return WriteAt(buffer_offset_, value);
  }

  absl::Status Write16BE(uint16_t value) {
    return Write16BEAt(buffer_offset_, value);
  }

  absl::Status Write16LE(uint16_t value) {
    return Write16LEAt(buffer_offset_, value);
  }

  absl::Status Write32BE(uint32_t value) {
    return Write32BEAt(buffer_offset_, value);
  }

  absl::Status Write32LE(uint32_t value) {
    return Write32LEAt(buffer_offset_, value);
  }

  absl::Status Write64BE(uint64_t value) {
    return Write64BEAt(buffer_offset_, value);
  }

  absl::Status Write64LE(uint64_t value) {
    return Write64LEAt(buffer_offset_, value);
  }

  template <typename T>
  absl::Status WriteMany(const T& value) {
    return Write(value);
  }

  template <typename T, typename... Args>
  absl::Status WriteMany(const T& value, const Args&... args) {
    TF_RETURN_IF_ERROR(Write(value));
    return WriteMany(args...);
  }

  // Returns an error if the buffer is not growable and writing `size` bytes at
  // `buffer_offset` would exceed `buffer_len_`.
  absl::Status WriteAt(size_t buffer_offset, const uint8_t* ptr, size_t size);

  absl::Status Write16BEAt(size_t buffer_offset, uint16_t ptr);
  absl::Status Write16LEAt(size_t buffer_offset, uint16_t ptr);
  absl::Status Write32BEAt(size_t buffer_offset, uint32_t ptr);
  absl::Status Write32LEAt(size_t buffer_offset, uint32_t ptr);
  absl::Status Write64BEAt(size_t buffer_offset, uint64_t ptr);
  absl::Status Write64LEAt(size_t buffer_offset, uint64_t ptr);

#define DEFINE_WRITE_AT(bytes, bits, type)                                     \
  template <typename T, std::enable_if_t<internal::IsBuiltinSerde<T>::value && \
                                         (sizeof(T) == bytes)>* = nullptr>     \
  absl::Status WriteAt(size_t buffer_offset, T value) {                        \
    switch (endian_) {                                                         \
      case Endian::kBig:                                                       \
        return Write##bits##BEAt(buffer_offset, value);                        \
      case Endian::kLittle:                                                    \
        return Write##bits##LEAt(buffer_offset, value);                        \
      case Endian::kNative:                                                    \
        return WriteAt(buffer_offset,                                          \
                       reinterpret_cast<const uint8_t*>(&value), bytes);       \
    }                                                                          \
    ABSL_UNREACHABLE();                                                        \
    return absl::InternalError("Corrupted endian");                            \
  }

  DEFINE_WRITE_AT(2, 16, uint16_t)
  DEFINE_WRITE_AT(4, 32, uint32_t)
  DEFINE_WRITE_AT(8, 64, uint64_t)
#undef DEFINE_WRITE_AT

  template <typename T,
            std::enable_if_t<internal::IsBuiltinSerde<T>::value &&
                             !((sizeof(T) == 2) || (sizeof(T) == 4) ||
                               (sizeof(T) == 8))>* = nullptr>
  absl::Status WriteAt(size_t buffer_offset, T value) {
    return WriteAt(buffer_offset, reinterpret_cast<const uint8_t*>(&value),
                   sizeof(T));
  }

  template <typename T,
            std::enable_if_t<internal::IsNonBuiltinSerde<T>::value>* = nullptr>
  absl::Status WriteAt(size_t buffer_offset, const T& value) {
    buffer_offset_ = buffer_offset;
    return Serde<T>::WriteTo(value, this, endian_);
  }

  template <typename T>
  absl::Status WriteManyAt(size_t buffer_offset, const T& value) {
    return WriteAt(buffer_offset, value);
  }

  template <typename T, typename... Args>
  absl::Status WriteManyAt(size_t buffer_offset, const T& value,
                           const Args&... args) {
    TF_RETURN_IF_ERROR(WriteAt(buffer_offset, value));
    return WriteManyAt(buffer_offset, args...);
  }

  virtual absl::Status Grow(size_t size) {
    return absl::UnimplementedError("Grow is not implemented");
  }
};

}  // namespace zkx::base

#endif  // ZKX_BASE_BUFFER_BUFFER_H_
