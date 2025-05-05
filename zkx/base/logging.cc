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

#include "zkx/base/logging.h"

#include <stddef.h>

#include "absl/strings/numbers.h"
#include "absl/synchronization/mutex.h"

#include "xla/tsl/platform/platform.h"

#if defined(PLATFORM_POSIX_ANDROID)
#include <iostream>
#include <sstream>

#include <android/log.h>
#else
#include <stdio.h>

#include <limits>

#include "absl/base/internal/sysinfo.h"
#include "absl/strings/str_format.h"

#include "xla/tsl/platform/env_time.h"
#endif

namespace zkx::base {
namespace internal {
namespace {

// This is an internal singleton class that manages the log sinks. It allows
// adding and removing the log sinks, as well as handling sending log messages
// to all the registered log sinks.
class LogSinks {
 public:
  // Gets the LogSinks instance. This is the entry point for using this class.
  static LogSinks& Instance();

  // Adds a log sink. The sink argument must not be a nullptr. LogSinks
  // takes ownership of the pointer, the user must not free the pointer.
  // The pointer will remain valid until the application terminates or
  // until LogSinks::Remove is called for the same pointer value.
  void Add(LogSink* sink);

  // Removes a log sink. This will also erase the sink object. The pointer
  // to the sink becomes invalid after this call.
  void Remove(LogSink* sink);

  // Gets the currently registered log sinks.
  std::vector<LogSink*> GetSinks() const;

  // Sends a log message to all registered log sinks.
  //
  // If there are no log sinks are registered:
  //
  // NO_DEFAULT_LOGGER is defined:
  // Up to 128 messages will be queued until a log sink is added.
  // The queue will then be logged to the first added log sink.
  //
  // NO_DEFAULT_LOGGER is not defined:
  // The messages will be logged using the default logger. The default logger
  // will log to stdout on all platforms except for Android. On Android, the
  // default Android logger will be used.
  void Send(const LogEntry& entry);

 private:
  LogSinks();
  void SendToSink(LogSink& sink, const LogEntry& entry);

  std::queue<LogEntry> log_entry_queue_;
  static const size_t kMaxLogEntryQueueSize = 128;

  mutable absl::Mutex mutex_;
  std::vector<LogSink*> sinks_;
};

LogSinks::LogSinks() {
#ifndef NO_DEFAULT_LOGGER
  static DefaultLogSink* default_sink = new DefaultLogSink();
  sinks_.push_back(default_sink);
#endif
}

LogSinks& LogSinks::Instance() {
  static LogSinks* instance = new LogSinks();
  return *instance;
}

void LogSinks::Add(LogSink* sink) {
  assert(sink != nullptr && "The sink must not be a nullptr");

  absl::MutexLock lock(&mutex_);
  sinks_.push_back(sink);

  // If this is the only sink log all the queued up messages to this sink
  if (sinks_.size() == 1) {
    while (!log_entry_queue_.empty()) {
      for (const auto& sink : sinks_) {
        SendToSink(*sink, log_entry_queue_.front());
      }
      log_entry_queue_.pop();
    }
  }
}

void LogSinks::Remove(LogSink* sink) {
  assert(sink != nullptr && "The sink must not be a nullptr");

  absl::MutexLock lock(&mutex_);
  auto it = std::find(sinks_.begin(), sinks_.end(), sink);
  if (it != sinks_.end()) sinks_.erase(it);
}

std::vector<LogSink*> LogSinks::GetSinks() const {
  absl::MutexLock lock(&mutex_);
  return sinks_;
}

void LogSinks::Send(const LogEntry& entry) {
  absl::MutexLock lock(&mutex_);

  // If we don't have any sinks registered, queue them up
  if (sinks_.empty()) {
    // If we've exceeded the maximum queue size, drop the oldest entries
    while (log_entry_queue_.size() >= kMaxLogEntryQueueSize) {
      log_entry_queue_.pop();
    }
    log_entry_queue_.push(entry);
    return;
  }

  // If we have items in the queue, push them out first
  while (!log_entry_queue_.empty()) {
    for (LogSink* sink : sinks_) {
      SendToSink(*sink, log_entry_queue_.front());
    }
    log_entry_queue_.pop();
  }

  // ... and now we can log the current log entry
  for (LogSink* sink : sinks_) {
    SendToSink(*sink, entry);
  }
}

void LogSinks::SendToSink(LogSink& sink, const LogEntry& entry) {
  sink.Send(entry);
  sink.WaitTillSent();
}

#ifndef PLATFORM_POSIX_ANDROID
// A class for managing the text file to which VLOG output is written.
// If the environment variable TF_CPP_VLOG_FILENAME is set, all VLOG
// calls are redirected from stderr to a file with corresponding name.
class VlogFileMgr {
 public:
  // Determines if the env variable is set and if necessary
  // opens the file for write access.
  VlogFileMgr();
  // Closes the file.
  ~VlogFileMgr();
  // Returns either a pointer to the file or stderr.
  FILE* FilePtr() const;

 private:
  FILE* vlog_file_ptr;
  char* vlog_file_name;
};

VlogFileMgr::VlogFileMgr() {
  vlog_file_name = getenv("ZKX_CPP_VLOG_FILENAME");
  vlog_file_ptr =
      vlog_file_name == nullptr ? nullptr : fopen(vlog_file_name, "w");

  if (vlog_file_ptr == nullptr) {
    vlog_file_ptr = stderr;
  }
}

VlogFileMgr::~VlogFileMgr() {
  if (vlog_file_ptr != stderr) {
    fclose(vlog_file_ptr);
  }
}

FILE* VlogFileMgr::FilePtr() const { return vlog_file_ptr; }

int ParseInteger(std::string_view str) {
  int level;
  if (!absl::SimpleAtoi(str, &level)) {
    return 0;
  }
  return level;
}

bool EmitThreadIdFromEnv() {
  const char* env_var_val = getenv("ZKX_CPP_LOG_THREAD_ID");
  return env_var_val == nullptr ? false : ParseInteger(env_var_val) != 0;
}

#endif  //  PLATFORM_POSIX_ANDROID

}  // namespace
}  // namespace internal

void AddLogSink(LogSink* sink) { internal::LogSinks::Instance().Add(sink); }

void RemoveLogSink(LogSink* sink) {
  internal::LogSinks::Instance().Remove(sink);
}

std::vector<LogSink*> GetLogSinks() {
  return internal::LogSinks::Instance().GetSinks();
}

void DefaultLogSink::Send(const LogEntry& entry) {
#ifdef PLATFORM_POSIX_ANDROID
  int android_log_level;
  switch (entry.log_severity()) {
    case absl::LogSeverity::kInfo:
      android_log_level = ANDROID_LOG_INFO;
      break;
    case absl::LogSeverity::kWarning:
      android_log_level = ANDROID_LOG_WARN;
      break;
    case absl::LogSeverity::kError:
      android_log_level = ANDROID_LOG_ERROR;
      break;
    case absl::LogSeverity::kFatal:
      android_log_level = ANDROID_LOG_FATAL;
      break;
    default:
      if (entry.log_severity() < absl::LogSeverity::kInfo) {
        android_log_level = ANDROID_LOG_VERBOSE;
      } else {
        android_log_level = ANDROID_LOG_ERROR;
      }
      break;
  }

  std::stringstream ss;
  const auto& fname = entry.FName();
  auto pos = fname.find("/");
  ss << (pos != std::string::npos ? fname.substr(pos + 1) : fname) << ":"
     << entry.Line() << " " << entry.ToString();
  __android_log_write(android_log_level, "native", ss.str().c_str());

  // Also log to stderr (for standalone Android apps).
  // Don't use 'std::cerr' since it crashes on Android.
  fprintf(stderr, "native : %s\n", ss.str().c_str());

  // Android logging at level FATAL does not terminate execution, so abort()
  // is still required to stop the program.
  if (entry.log_severity() == absl::LogSeverity::kFatal) {
    abort();
  }
#else  // PLATFORM_POSIX_ANDROID
  static const internal::VlogFileMgr vlog_file;
  static bool log_thread_id = internal::EmitThreadIdFromEnv();
  uint64_t now_micros = tsl::EnvTime::NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
  const size_t time_buffer_size = 30;
  char time_buffer[time_buffer_size];
  struct tm* tp;
#if defined(__linux__) || defined(__APPLE__)
  struct tm now_tm;
  tp = localtime_r(&now_seconds, &now_tm);
#else
  tp = localtime(&now_seconds);  // NOLINT(runtime/threadsafe_fn)
#endif
  strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S", tp);
  uint64_t tid = absl::base_internal::GetTID();
  constexpr size_t kTidBufferSize =
      (1 + std::numeric_limits<uint64_t>::digits10 + 1);
  char tid_buffer[kTidBufferSize] = "";
  if (log_thread_id) {
    absl::SNPrintF(tid_buffer, sizeof(tid_buffer), " %7u", tid);
  }

  char sev;
  switch (entry.log_severity()) {
    case absl::LogSeverity::kInfo:
      sev = 'I';
      break;

    case absl::LogSeverity::kWarning:
      sev = 'W';
      break;

    case absl::LogSeverity::kError:
      sev = 'E';
      break;

    case absl::LogSeverity::kFatal:
      sev = 'F';
      break;

    default:
      assert(false && "Unknown logging severity");
      sev = '?';
      break;
  }

  absl::FPrintF(vlog_file.FilePtr(), "%s.%06d: %c%s %s:%d] %s\n", time_buffer,
                micros_remainder, sev, tid_buffer, entry.FName().c_str(),
                entry.Line(), entry.ToString().c_str());
  fflush(vlog_file.FilePtr());  // Ensure logs are written immediately.
#endif  // PLATFORM_POSIX_ANDROID
}

}  // namespace zkx::base
