/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_TSL_PLATFORM_ENV_H_
#define XLA_TSL_PLATFORM_ENV_H_

#include <stddef.h>
#include <stdint.h>

#include <string_view>

#include "absl/base/attributes.h"
#include "absl/functional/any_invocable.h"

#include "xla/tsl/platform/env_time.h"

namespace tsl {

class Thread;
struct ThreadOptions;

// NOTE(chokobole): This class includes portions of the Env class to enable
// zkx::thread::ThreadPool and zkx::cpu::ThunkExecutor from:
// https://github.com/openxla/xla/blob/8bac4a2/xla/tsl/platform/env.h.
// It is planned to be replaced with Chromium's implementation in the future.
class Env {
 public:
  Env();
  virtual ~Env() = default;

  /// \brief Returns a default environment suitable for the current operating
  /// system.
  ///
  /// Sophisticated users may wish to provide their own Env
  /// implementation instead of relying on this default environment.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static Env* Default();

  /// \brief Returns the number of nano-seconds since the Unix epoch.
  virtual uint64_t NowNanos() const { return EnvTime::NowNanos(); }

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  virtual uint64_t NowMicros() const { return EnvTime::NowMicros(); }

  /// \brief Returns the number of seconds since the Unix epoch.
  virtual uint64_t NowSeconds() const { return EnvTime::NowSeconds(); }

  /// \brief Returns a new thread that is running fn() and is identified
  /// (for debugging/performance-analysis) by "name".
  ///
  /// Caller takes ownership of the result and must delete it eventually
  /// (the deletion will block until fn() stops running).
  virtual Thread* StartThread(
      const ThreadOptions& thread_options, std::string_view name,
      absl::AnyInvocable<void()> fn) ABSL_MUST_USE_RESULT = 0;
};

// NOTE(chokobole): It is planned to be replaced with Chromium's implementation
// in the future.
/// Represents a thread used to run a TSL function.
class Thread {
 public:
  Thread() {}

  /// Blocks until the thread of control stops running.
  virtual ~Thread();

 private:
  Thread(const Thread&) = delete;
  void operator=(const Thread&) = delete;
};

// NOTE(chokobole): It is planned to be replaced with Chromium's implementation
// in the future.
/// \brief Options to configure a Thread.
///
/// Note that the options are all hints, and the
/// underlying implementation may choose to ignore it.
struct ThreadOptions {
  /// Thread stack size to use (in bytes).
  size_t stack_size = 0;  // 0: use system default value
  /// Guard area size to use near thread stacks to use (in bytes)
  size_t guard_size = 0;  // 0: use system default value
  // TODO(chokobole): Uncomment this. Dependency: numa
  // int numa_node = port::kNUMANoAffinity;
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_ENV_H_
