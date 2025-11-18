/* Copyright 2019 The OpenXLA Authors.
Copyright 2025 The ZKX Authors.

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

#include "zkx/service/slow_operation_alarm.h"

#include <list>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"

#include "xla/tsl/platform/env.h"

namespace zkx {
namespace {

absl::Mutex mu(absl::kConstInit);
absl::CondVar* ready;
absl::once_flag init_flag;
std::list<SlowOperationAlarm*>* outstanding_alarms ABSL_PT_GUARDED_BY(mu) =
    nullptr;

}  // namespace

// static
void SlowOperationAlarm::AlarmLoop() {
  while (true) {
    absl::MutexLock lock(&mu);

    // Fire any alarms which are ready.
    absl::Time now = absl::Now();
    for (auto it = outstanding_alarms->begin();
         it != outstanding_alarms->end();) {
      auto next = std::next(it);
      auto* alarm = *it;
      // Fire the alarm if applicable.
      if (alarm->deadline() <= now) {
        outstanding_alarms->erase(it);
        const int64_t count =
            alarm->counter() == nullptr ? 0 : alarm->counter()->fetch_add(1);
        // If the alarm has a counter, only fire if the count is a power of 2.
        if (count == 0 || absl::has_single_bit<uint64_t>(count)) {
          alarm->fired_.store(true);
          // We fire alarms with LOG(ERROR) because otherwise it might not show
          // up without --logtostderr.
          alarm->msg_ = alarm->msg_fn_();
          LOG(ERROR) << alarm->msg_;
        }
      }
      it = next;
    }

    auto next_alarm = absl::c_min_element(
        *outstanding_alarms,
        [](const SlowOperationAlarm* a, const SlowOperationAlarm* b) -> bool {
          return a->deadline() < b->deadline();
        });
    const absl::Time deadline = next_alarm != outstanding_alarms->end()
                                    ? (*next_alarm)->deadline()
                                    : absl::InfiniteFuture();

    ready->WaitWithDeadline(&mu, deadline);
  }
}

// static
void SlowOperationAlarm::ScheduleAlarm(SlowOperationAlarm* alarm) {
  absl::call_once(init_flag, [] {
    ready = absl::IgnoreLeak(new absl::CondVar());
    outstanding_alarms = absl::IgnoreLeak(new std::list<SlowOperationAlarm*>());
    [[maybe_unused]] static tsl::Thread* t =
        absl::IgnoreLeak(tsl::Env::Default()->StartThread(
            tsl::ThreadOptions(), "SlowOperationAlarm", [] { AlarmLoop(); }));
  });

  absl::MutexLock lock(&mu);
  outstanding_alarms->push_back(alarm);
  ready->Signal();
}

// static
void SlowOperationAlarm::UnscheduleAlarm(const SlowOperationAlarm* alarm) {
  absl::MutexLock lock(&mu);
  CHECK_NE(outstanding_alarms, nullptr);
  auto it = absl::c_find(*outstanding_alarms, alarm);
  if (it != outstanding_alarms->end()) {
    outstanding_alarms->erase(it);
  }
}

SlowOperationAlarm::SlowOperationAlarm(
    absl::Duration timeout, std::string msg,
    std::atomic<int64_t>* counter /*=nullptr*/,
    std::string_view context /*=""*/)
    : SlowOperationAlarm(
          timeout, [msg = std::move(msg)] { return msg; }, counter,
          std::move(context)) {}

SlowOperationAlarm::SlowOperationAlarm(
    absl::Duration timeout, std::function<std::string()> msg_fn,
    std::atomic<int64_t>* counter /*=nullptr*/,
    std::string_view context /*=""*/)
    : start_(absl::Now()),
      deadline_(start_ + timeout),
      context_(std::move(context)),
      msg_fn_(std::move(msg_fn)),
      counter_(counter) {
  ScheduleAlarm(this);
}

SlowOperationAlarm::~SlowOperationAlarm() {
  UnscheduleAlarm(this);

  if (fired()) {
    absl::Duration duration = absl::Now() - start_;
    if (context_.empty()) {
      LOG(ERROR) << "The operation took " << absl::FormatDuration(duration)
                 << "\n"
                 << msg_;
    } else {
      LOG(ERROR) << "[" << context_ << "] The operation took "
                 << absl::FormatDuration(duration) << "\n"
                 << msg_;
    }
  }
}

std::unique_ptr<SlowOperationAlarm> SlowCompilationAlarm(
    std::string_view context) {
  // Pass a counter to these alarms so they only log once every power-of-two
  // occurrences.
  static auto* counter = absl::IgnoreLeak(new std::atomic<int64_t>(0));

  const char* separator = "\n********************************";

  std::string context_msg;
  if (!context.empty()) {
    context_msg = absl::StrCat("[", context, "] ");
  }

#if NDEBUG
  return std::make_unique<SlowOperationAlarm>(
      absl::Duration(absl::Minutes(2)),
      absl::StrCat(
          separator, "\n", context_msg,
          "Very slow compile? If you want to file a bug, run with envvar "
          "ZKX_FLAGS=--zkx_dump_to=/tmp/foo and attach the results.",
          separator),
      counter);
#else
  return std::make_unique<SlowOperationAlarm>(
      absl::Duration(absl::Seconds(10)),
      absl::StrCat(
          separator, "\n", context_msg,
          "Slow compile? ZKX was built without compiler optimizations, which "
          "can be slow. Try rebuilding with -c opt.",
          separator),
      counter);
#endif
}

}  // namespace zkx
