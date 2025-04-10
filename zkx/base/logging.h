#ifndef ZKX_BASE_LOGGING_H_
#define ZKX_BASE_LOGGING_H_

#include "absl/log/log.h"

#define VLOG(level) LOG(INFO).WithVerbosity(level)

#endif  // ZKX_BASE_LOGGING_H_
