/* Copyright 2017 The OpenXLA Authors.

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

// The compiler API is used by the XLA service to generate executables that
// run on a given platform. This is a registry and abstract interface, for
// pluggability by the various platforms.

#ifndef ZKX_SERVICE_COMPILER_H_
#define ZKX_SERVICE_COMPILER_H_

#include <functional>
#include <memory>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

#include "zkx/stream_executor/platform.h"

namespace zkx {

// Abstract compiler interface that is subclassed for compilation on a
// particular platform.
//
// The compiler ties together high level optimization (HLO) and low level
// optimization (LLO) / codegen (CG) to generate efficient executables for the
// target platform.
//
// The platform-based compiler singletons are registered via module initializers
// in their corresponding ZKX compiler libraries, and are registered via the
// RegisterCompilerFactory API below.
//
// Thread-safety: subclasses of Compiler must be thread-safe, as multiple
// ZKX clients may be requesting compilation concurrently for a given
// platform.
class Compiler {
 public:
  // The Compiler class also serves as a point to register compiler objects
  // for the various platforms.

  using CompilerFactory = std::function<std::unique_ptr<Compiler>()>;

  // Registers the compiler singleton for the platform. This is assumed to
  // be a singleton, so no ownership is transferred.
  //
  // Precondition: a platform kind must not be registered more than once.
  static void RegisterCompilerFactory(se::Platform::Id platform_id,
                                      CompilerFactory compiler_factory);

  // Returns the compiler singleton pointer if it is available for the given
  // platform, or an error status if it is not.
  static absl::StatusOr<Compiler*> GetForPlatform(const se::Platform* platform);

 private:
  // Mutex that guards the platform-compiler map.
  ABSL_CONST_INIT static absl::Mutex platform_compiler_mutex_;

  // Map from platform kind to compiler factory.
  static absl::flat_hash_map<se::Platform::Id, CompilerFactory>*
  GetPlatformCompilerFactories();

  // Map from platform kind to compiler instance, if we made one already (based
  // on the factories above).
  static absl::flat_hash_map<se::Platform::Id, std::unique_ptr<Compiler>>*
  GetPlatformCompilers();
};

}  // namespace zkx

#endif  // ZKX_SERVICE_COMPILER_H_
