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

#include <map>
#include <string>
#include <thread>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"

#include "xla/tsl/platform/env.h"

namespace tsl {

absl::Mutex g_name_mutex;

std::map<std::thread::id, std::string>& GetThreadNameRegistry()
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(g_name_mutex) {
  static auto* thread_name_registry =
      new std::map<std::thread::id, std::string>();
  return *thread_name_registry;
}

// We use the pthread API instead of std::thread so we can control stack sizes.
class PThread : public Thread {
 public:
  PThread(const ThreadOptions& thread_options, std::string_view name,
          absl::AnyInvocable<void()> fn) {
    ThreadParams* params = new ThreadParams;
    params->name = std::move(name);
    params->fn = std::move(fn);
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    if (thread_options.stack_size != 0) {
      pthread_attr_setstacksize(&attributes, thread_options.stack_size);
    }
    int ret = pthread_create(&thread_, &attributes, &ThreadFn, params);
    // There is no mechanism for the thread creation API to fail, so we CHECK.
    CHECK_EQ(ret, 0) << "Thread " << name
                     << " creation via pthread_create() failed.";
    pthread_attr_destroy(&attributes);
  }

  ~PThread() override { pthread_join(thread_, nullptr); }

 private:
  struct ThreadParams {
    std::string name;
    absl::AnyInvocable<void()> fn;
  };
  static void* ThreadFn(void* params_arg) {
    std::unique_ptr<ThreadParams> params(
        reinterpret_cast<ThreadParams*>(params_arg));
    {
      absl::MutexLock l(&g_name_mutex);
      GetThreadNameRegistry().emplace(std::this_thread::get_id(), params->name);
    }
    params->fn();
    {
      absl::MutexLock l(&g_name_mutex);
      GetThreadNameRegistry().erase(std::this_thread::get_id());
    }
    return nullptr;
  }

  pthread_t thread_;
};

class PosixEnv : public Env {
 public:
  PosixEnv() {}

  ~PosixEnv() override { LOG(FATAL) << "Env::Default() must not be destroyed"; }

  Thread* StartThread(const ThreadOptions& thread_options,
                      std::string_view name,
                      absl::AnyInvocable<void()> fn) override {
    return new PThread(thread_options, name, std::move(fn));
  }
};

Env* Env::Default() {
  static Env* default_env = new PosixEnv;
  return default_env;
}

}  // namespace tsl
