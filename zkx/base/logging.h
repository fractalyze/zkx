#ifndef ZKX_BASE_LOGGING_H_
#define ZKX_BASE_LOGGING_H_

#include "absl/log/log.h"

namespace zkx::base {

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
