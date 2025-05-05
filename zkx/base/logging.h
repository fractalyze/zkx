#ifndef ZKX_BASE_LOGGING_H_
#define ZKX_BASE_LOGGING_H_

#include <string>

#include "absl/log/log.h"

namespace zkx::base {

// LogSink support adapted from absl/log/log.h
//
// `LogSink` is an interface which can be extended to intercept and process
// all log messages. LogSink implementations must be thread-safe. A single
// instance will be called from whichever thread is performing a logging
// operation.
class LogEntry {
 public:
  explicit LogEntry(absl::LogSeverity severity, std::string_view message)
      : severity_(severity), message_(message) {}

  explicit LogEntry(absl::LogSeverity severity, std::string_view fname,
                    int line, std::string_view message)
      : severity_(severity), fname_(fname), line_(line), message_(message) {}

  absl::LogSeverity log_severity() const { return severity_; }
  std::string FName() const { return fname_; }
  int Line() const { return line_; }
  std::string ToString() const { return message_; }
  std::string_view text_message() const { return message_; }

  // Returning similar result as `text_message` as there is no prefix in this
  // implementation.
  std::string_view text_message_with_prefix() const { return message_; }

 private:
  const absl::LogSeverity severity_;
  const std::string fname_;
  int line_ = -1;
  const std::string message_;
};

class LogSink {
 public:
  virtual ~LogSink() = default;

  // `Send` is called synchronously during the log statement.  The logging
  // module guarantees not to call `Send` concurrently on the same log sink.
  // Implementations should be careful not to call`LOG` or `CHECK` or take
  // any locks that might be held by the `LOG` caller, to avoid deadlock.
  //
  // `e` is guaranteed to remain valid until the subsequent call to
  // `WaitTillSent` completes, so implementations may store a pointer to or
  // copy of `e` (e.g. in a thread local variable) for use in `WaitTillSent`.
  virtual void Send(const LogEntry& entry) = 0;

  // `WaitTillSent` blocks the calling thread (the thread that generated a log
  // message) until the sink has finished processing the log message.
  // `WaitTillSent` is called once per log message, following the call to
  // `Send`. This may be useful when log messages are buffered or processed
  // asynchronously by an expensive log sink.
  // The default implementation returns immediately. Like `Send`,
  // implementations should be careful not to call `LOG` or `CHECK or take any
  // locks that might be held by the `LOG` caller, to avoid deadlock.
  virtual void WaitTillSent() {}
};

// This is the default log sink. This log sink is used if there are no other
// log sinks registered. To disable the default log sink, set the
// "no_default_logger" Bazel config setting to true or define a
// NO_DEFAULT_LOGGER preprocessor symbol. This log sink will always log to
// stderr.
class DefaultLogSink : public LogSink {
 public:
  void Send(const LogEntry& entry) override;
};

// Add or remove a `LogSink` as a consumer of logging data. Thread-safe.
void AddLogSink(LogSink* sink);
void RemoveLogSink(LogSink* sink);

// Get all the log sinks. Thread-safe.
std::vector<LogSink*> GetLogSinks();

template <typename T>
T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t) {
  if (t == nullptr) {
    absl::log_internal::LogMessageFatal(file, line) << exprtext;
  }
  return std::forward<T>(t);
}

}  // namespace zkx::base

#define CHECK_NOTNULL(val)                                                     \
  ::zkx::base::CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", \
                            (val))

#define VLOG(level) LOG(INFO).WithVerbosity(level)

#endif  // ZKX_BASE_LOGGING_H_
