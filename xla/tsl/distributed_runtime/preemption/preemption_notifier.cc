/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/tsl/distributed_runtime/preemption/preemption_notifier.h"

#include <atomic>
#include <csignal>

#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"

namespace tsl {

namespace {

constexpr absl::Duration kListenInterval = absl::Seconds(1);
constexpr absl::Time kUnsetDeathTime = absl::InfinitePast();
static std::atomic_bool g_sigterm_received(false);

class SigtermNotifier : public PreemptionNotifier {
 public:
  explicit SigtermNotifier(Env* env);
  ~SigtermNotifier() override {
    // Trigger shutdown logic in listener thread.
    shutdown_notification_.Notify();
  }

 private:
  void StartListenerThread();
  absl::Notification shutdown_notification_;
  std::unique_ptr<Thread> preempt_listener_thread_;
};

SigtermNotifier::SigtermNotifier(Env* env) : PreemptionNotifier(env) {
  g_sigterm_received.store(false);
  StartListenerThread();
#if defined(PLATFORM_GOOGLE)
  thread::signal::Token unused_token;

  thread::signal::AddHandler(
      SIGTERM, thread::Executor::DefaultExecutor(),
      []() { g_sigterm_received.store(true); },
      /*flags=*/0,  // Don't override existing signal handlers.
      &unused_token);
#else
  std::signal(SIGTERM, [](int signal) { g_sigterm_received.store(true); });
#endif
}

void SigtermNotifier::StartListenerThread() {
  preempt_listener_thread_.reset(
      GetEnv()->StartThread({}, "PreemptionNotifier_Listen", [this]() {
        // Poll for SIGTERM receipt every kListenInterval.
        while (!g_sigterm_received.load()) {
          if (shutdown_notification_.WaitForNotificationWithTimeout(
                  kListenInterval)) {
            // Shutdown:
            // 1) Cancel any pending callbacks and blocking WillBePreemptedAt()
            // calls.
            NotifyRegisteredListeners(
                absl::CancelledError("Preemption notifier is being deleted."));
            // 2) Exit listener thread.
            return;
          }
        }
        const absl::Time death_time = absl::Now();
        LOG(WARNING) << "SIGTERM caught at " << death_time;
        // Notify registered listeners.
        NotifyRegisteredListeners(death_time);
      }));
}

}  // namespace

absl::StatusOr<absl::Time> PreemptionNotifier::WillBePreemptedAt() {
  absl::Notification n;
  absl::StatusOr<absl::Time> result;
  WillBePreemptedAtAsync(
      [&n, &result](absl::StatusOr<absl::Time> async_result) {
        result = async_result;
        n.Notify();
      });
  n.WaitForNotification();
  return result;
}

void PreemptionNotifier::WillBePreemptedAtAsync(PreemptTimeCallback callback) {
  absl::MutexLock l(&mu_);
  if (death_time_ == kUnsetDeathTime) {
    // Did not receive preemption notice yet.
    callbacks_.push_back(std::move(callback));
  } else {
    // Already received preemption notice, respond immediately.
    callback(death_time_);
  }
}

void PreemptionNotifier::NotifyRegisteredListeners(
    absl::StatusOr<absl::Time> death_time) {
  absl::MutexLock l(&mu_);
  if (death_time.ok()) {
    death_time_ = death_time.value();
  }
  for (const PreemptTimeCallback& callback : callbacks_) {
    callback(death_time);
  }
  callbacks_.clear();
}

REGISTER_PREEMPTION_NOTIFIER(
    "sigterm", [](Env* env) -> std::unique_ptr<PreemptionNotifier> {
      return std::make_unique<SigtermNotifier>(env);
    });
}  // namespace tsl
