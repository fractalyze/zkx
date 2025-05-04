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
#ifndef ZKX_SERVICE_CUSTOM_CALL_TARGET_REGISTRY_H_
#define ZKX_SERVICE_CUSTOM_CALL_TARGET_REGISTRY_H_

// This file is depended on by kernels that have to build for mobile devices.
// For this reason, we avoid relying on TensorFlow and instead only use the
// standard C++ library.

#include <stddef.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace zkx {

// ZKX JIT compilers use this registry to resolve symbolic CustomCall targets;
// so when using ZKX as a JIT, CustomCall targets need to be registered here
// with the symbol name used in the CustomCall.
//
// The ZKX:CPU ahead-of-time (AOT) compiler links using a standard offline
// linker; so when compiling in CPU AOT mode, you *also* need to make sure the
// name of the callee (presumably implemented in C++) matches up with the
// symbolic name used in the CustomCall. Be careful with the name of the symbol
// you register with the macros: C++ namespaces are not included, including
// anonymous namespaces,so if two libraries attempt to register functions with
// the same name in separate namespaces the registrations will collide. Either
// call the registration macro from the global namespace so that you have to
// refer to the function in a fully-qualified manner (which also requires you to
// emit HLO-based calls to it by the fully-qualified name *and* complicates
// future refactoring!) or use C-style namespacing directly in the symbol name.
//
// We maintain the registry in both the JIT and the AOT cases for simplicity,
// but we only use it when running in JIT mode.
class CustomCallTargetRegistry {
 public:
  static CustomCallTargetRegistry* Global();

  void Register(const std::string& symbol, void* address,
                const std::string& platform);
  void* Lookup(const std::string& symbol, const std::string& platform) const;

  std::unordered_map<std::string, void*> registered_symbols(
      const std::string& platform) const;

 private:
  // hash<pair<T,T>> is surprisingly not provided by default in stl.  It would
  // be better to use absl's hash function, but we're avoiding an absl
  // dependency here because this library is pulled in by all ZKX:CPU AoT
  // binaries.
  struct HashPairOfStrings {
    size_t operator()(const std::pair<std::string, std::string>& k) const {
      std::hash<std::string> hasher;
      size_t h1 = hasher(k.first);
      size_t h2 = hasher(k.second);
      // This is a bad hash function, but it's good enough for our purposes
      // here.  Nobody is going to try to DoS this hashtable.  :)
      return h1 ^ 31 * h2;
    }
  };

  // Maps the pair (symbol, platform) to a C function implementing a custom call
  // named `symbol` for StreamExecutor platform `platform`.
  //
  // Different platforms have different ABIs.  TODO(jlebar): Describe them!
  //
  // (We use std::unordered_map and std::mutex rather than absl::flat_hash_map
  // and absl::mutex because we want to avoid an absl dependency, because this
  // library is pulled in by all ZKX:CPU AoT binaries.)
  std::unordered_map<std::pair<std::string, std::string>, void*,
                     HashPairOfStrings>
      registered_symbols_;
  mutable std::mutex mu_;
};

class RegisterCustomCallTarget {
 public:
  explicit RegisterCustomCallTarget(const std::string& name, void* address,
                                    const std::string& platform) {
    CustomCallTargetRegistry::Global()->Register(name, address, platform);
  }
};

#define ZKX_REGISTER_CUSTOM_CALL_CONCAT(a, b) a##b

#define ZKX_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM_HELPER(symbol, address,   \
                                                        platform, counter) \
  static ::zkx::RegisterCustomCallTarget ZKX_REGISTER_CUSTOM_CALL_CONCAT(  \
      custom_call_target_register, counter)(                               \
      symbol, reinterpret_cast<void*>(address), platform)

#define ZKX_REGISTER_CUSTOM_CALL_TARGET(function, platform) \
  ZKX_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(#function, function, platform)

#define ZKX_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(symbol, address, platform)  \
  ZKX_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM_HELPER(symbol, address, platform, \
                                                  __COUNTER__)

// Convenience overloads for registering custom-call targets on the CPU.
#define ZKX_CPU_REGISTER_CUSTOM_CALL_TARGET(function) \
  ZKX_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(#function, function, "Host")

#define ZKX_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(symbol, address) \
  ZKX_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(symbol, address, "Host")

}  // namespace zkx

#endif  // ZKX_SERVICE_CUSTOM_CALL_TARGET_REGISTRY_H_
