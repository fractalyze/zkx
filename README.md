# ZKX: Zero Knowledge Accelerator

## 🚀 Build Error Troubleshooting Guide

This document provides solutions for common build errors encountered when compiling this project.
If you encounter an issue that is not listed here, feel free to contribute by opening a PR or issue!

### 🛠️ Error: Aligned Deallocation on macOS

```shell
./xla/tsl/concurrency/async_value.h:1031:9: error: aligned deallocation function of type 'void (void *, std::align_val_t) noexcept' is only available on macOS 10.13 or newer
 1031 |       ::operator delete(this, std::align_val_t{alignof(IndirectAsyncValue)});
      |         ^
./xla/tsl/concurrency/async_value.h:1031:9: note: if you supply your own aligned allocation functions, use -faligned-allocation to silence this diagnostic
./xla/tsl/concurrency/async_value.h:1042:7: error: aligned deallocation function of type 'void (void *, std::align_val_t) noexcept' is only available on macOS 10.13 or newer
 1042 |     ::operator delete(this, alignment);
```

To ensure compatibility, add the following line to your `.bazelrc.user` file in the project root:

```shell
build --macos_minimum_os=10.13
```
