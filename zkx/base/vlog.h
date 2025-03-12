#ifndef ZX_BASE_VLOG_H_
#define ZX_BASE_VLOG_H_

#include "absl/log/log.h"

#define VLOG(level) LOG(INFO).WithVerbosity(level)

#endif  // ZX_BASE_VLOG_H_
