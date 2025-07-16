#ifndef ZKX_BASE_BUFFER_READ_ONLY_BUFFER_H_
#define ZKX_BASE_BUFFER_READ_ONLY_BUFFER_H_

#include <stddef.h>
#include <stdint.h>

#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"

#include "xla/tsl/platform/errors.h"
#include "zkx/base/buffer/endian.h"
#include "zkx/base/buffer/serde_forward.h"

namespace zkx::base {
namespace internal {

template <typename, typename = void>
struct IsBuiltinSerde : std::false_type {};

template <typename T>
struct IsBuiltinSerde<T, std::enable_if_t<std::is_fundamental_v<T>>>
    : std::true_type {};

template <typename, typename = void>
struct IsNonBuiltinSerde : std::false_type {};

template <typename T>
struct IsNonBuiltinSerde<
    T, std::enable_if_t<!IsBuiltinSerde<T>::value && IsSerde<T>::value>>
    : std::true_type {};

}  // namespace internal

class ReadOnlyBuffer {
 public:
  ReadOnlyBuffer() = default;
  ReadOnlyBuffer(const void* buffer, size_t buffer_len)
      : buffer_(const_cast<void*>(buffer)),
        buffer_offset_(0),
        buffer_len_(buffer_len) {}
  ReadOnlyBuffer(const ReadOnlyBuffer& other) = delete;
  ReadOnlyBuffer& operator=(const ReadOnlyBuffer& other) = delete;
  ReadOnlyBuffer(ReadOnlyBuffer&& other)
      : buffer_(std::exchange(other.buffer_, nullptr)),
        buffer_offset_(std::exchange(other.buffer_offset_, 0)),
        buffer_len_(std::exchange(other.buffer_len_, 0)) {}
  ReadOnlyBuffer& operator=(ReadOnlyBuffer&& other) {
    buffer_ = std::exchange(other.buffer_, nullptr);
    buffer_offset_ = std::exchange(other.buffer_offset_, 0);
    buffer_len_ = std::exchange(other.buffer_len_, 0);
    return *this;
  }
  ~ReadOnlyBuffer() = default;

  Endian endian() const { return endian_; }
  // NOTE(chokobole): Marked as const to allow calls on `const ReadOnlyBuffer&`,
  // even though it mutates internal state.
  void set_endian(Endian endian) const { endian_ = endian; }

  const void* buffer() const { return buffer_; }

  size_t buffer_offset() const { return buffer_offset_; }
  // NOTE(chokobole): Marked as const to allow calls on `const ReadOnlyBuffer&`,
  // even though it mutates internal state.
  void set_buffer_offset(size_t buffer_offset) const {
    buffer_offset_ = buffer_offset;
  }

  size_t buffer_len() const { return buffer_len_; }

  bool Done() const { return buffer_offset_ == buffer_len_; }

  // Returns an error if reading `size` bytes from `buffer_offset` would exceed
  // `buffer_len_`.
  absl::Status ReadAt(size_t buffer_offset, uint8_t* ptr, size_t size) const;

  template <typename Ptr, typename T = std::remove_pointer_t<Ptr>,
            std::enable_if_t<std::is_pointer_v<Ptr> &&
                             internal::IsBuiltinSerde<T>::value>* = nullptr>
  absl::Status ReadAt(size_t buffer_offset, Ptr ptr) const {
    switch (endian_) {
      case Endian::kBig:
        if constexpr (sizeof(T) == 8) {
          return Read64BEAt(buffer_offset, reinterpret_cast<uint64_t*>(ptr));
        } else if constexpr (sizeof(T) == 4) {
          return Read32BEAt(buffer_offset, reinterpret_cast<uint32_t*>(ptr));
        } else if constexpr (sizeof(T) == 2) {
          return Read16BEAt(buffer_offset, reinterpret_cast<uint16_t*>(ptr));
        }
      case Endian::kLittle:
        if constexpr (sizeof(T) == 8) {
          return Read64LEAt(buffer_offset, reinterpret_cast<uint64_t*>(ptr));
        } else if constexpr (sizeof(T) == 4) {
          return Read32LEAt(buffer_offset, reinterpret_cast<uint32_t*>(ptr));
        } else if constexpr (sizeof(T) == 2) {
          return Read16LEAt(buffer_offset, reinterpret_cast<uint16_t*>(ptr));
        }
      case Endian::kNative:
        return ReadAt(buffer_offset, reinterpret_cast<uint8_t*>(ptr),
                      sizeof(T));
    }
    ABSL_UNREACHABLE();
    return absl::InternalError("Corrupted endian");
  }

  absl::Status Read16BEAt(size_t buffer_offset, uint16_t* ptr) const;
  absl::Status Read16LEAt(size_t buffer_offset, uint16_t* ptr) const;
  absl::Status Read32BEAt(size_t buffer_offset, uint32_t* ptr) const;
  absl::Status Read32LEAt(size_t buffer_offset, uint32_t* ptr) const;
  absl::Status Read64BEAt(size_t buffer_offset, uint64_t* ptr) const;
  absl::Status Read64LEAt(size_t buffer_offset, uint64_t* ptr) const;

  template <typename T,
            std::enable_if_t<internal::IsNonBuiltinSerde<T>::value>* = nullptr>
  absl::Status ReadAt(size_t buffer_offset, T* value) const {
    buffer_offset_ = buffer_offset;
    return Serde<T>::ReadFrom(*this, value, endian_);
  }

  template <typename T, size_t N>
  absl::Status ReadAt(size_t buffer_offset, T (&array)[N]) const {
    buffer_offset_ = buffer_offset;
    for (size_t i = 0; i < N; ++i) {
      TF_RETURN_IF_ERROR(Read(&array[i]));
    }
    return absl::OkStatus();
  }

  template <typename T>
  absl::Status ReadPtrAt(size_t buffer_offset, T** ptr, size_t ptr_num) const {
    size_t size = sizeof(T) * ptr_num;
    // TODO(chokobole): add overflow check
    if (buffer_offset + size > buffer_len_) {
      return absl::OutOfRangeError(
          absl::Substitute("Read out of bounds: buffer_offset ($0) + size ($1) "
                           "exceeds buffer_len_ ($2)",
                           buffer_offset, size, buffer_len_));
    }
    const char* buffer = reinterpret_cast<const char*>(buffer_);
    *ptr = const_cast<T*>(reinterpret_cast<const T*>(&buffer[buffer_offset]));
    buffer_offset_ = buffer_offset + size;
    return absl::OkStatus();
  }

  absl::Status Read(uint8_t* ptr, size_t size) const {
    return ReadAt(buffer_offset_, ptr, size);
  }

  template <typename T>
  absl::Status Read(T&& value) const {
    return ReadAt(buffer_offset_, std::forward<T>(value));
  }

  template <typename T>
  absl::Status ReadPtr(T** ptr, size_t ptr_num) const {
    return ReadPtrAt(buffer_offset_, ptr, ptr_num);
  }

  absl::Status Read16BE(uint16_t* ptr) const {
    return Read16BEAt(buffer_offset_, ptr);
  }

  absl::Status Read16LE(uint16_t* ptr) const {
    return Read16LEAt(buffer_offset_, ptr);
  }

  absl::Status Read32BE(uint32_t* ptr) const {
    return Read32BEAt(buffer_offset_, ptr);
  }

  absl::Status Read32LE(uint32_t* ptr) const {
    return Read32LEAt(buffer_offset_, ptr);
  }

  absl::Status Read64BE(uint64_t* ptr) const {
    return Read64BEAt(buffer_offset_, ptr);
  }

  absl::Status Read64LE(uint64_t* ptr) const {
    return Read64LEAt(buffer_offset_, ptr);
  }

  template <typename T>
  absl::Status ReadMany(T&& value) const {
    return Read(std::forward<T>(value));
  }

  template <typename T, typename... Args>
  absl::Status ReadMany(T&& value, Args&&... args) const {
    TF_RETURN_IF_ERROR(Read(std::forward<T>(value)));
    return ReadMany(std::forward<Args>(args)...);
  }

  template <typename T, size_t N>
  absl::Status ReadManyAt(size_t buffer_offset, T&& value) const {
    return ReadAt(buffer_offset, std::forward<T>(value));
  }

  template <typename T, typename... Args>
  absl::Status ReadManyAt(size_t buffer_offset, T&& value,
                          Args&&... args) const {
    TF_RETURN_IF_ERROR(ReadAt(buffer_offset, std::forward<T>(value)));
    return ReadManyAt(buffer_offset, std::forward<Args>(args)...);
  }

 protected:
  mutable Endian endian_ = Endian::kNative;

  void* buffer_ = nullptr;
  mutable size_t buffer_offset_ = 0;
  size_t buffer_len_ = 0;
};

}  // namespace zkx::base

#endif  // ZKX_BASE_BUFFER_READ_ONLY_BUFFER_H_
