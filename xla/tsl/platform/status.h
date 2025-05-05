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

#ifndef XLA_TSL_PLATFORM_STATUS_H_
#define XLA_TSL_PLATFORM_STATUS_H_

#include <functional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"

namespace tsl {

typedef std::function<void(const absl::Status&)> StatusCallback;

std::string* TfCheckOpHelperOutOfLine(const absl::Status& v, const char* msg);

inline std::string* TfCheckOpHelper(absl::Status v, const char* msg) {
  if (v.ok()) return nullptr;
  return TfCheckOpHelperOutOfLine(v, msg);
}

#define TF_DO_CHECK_OK(val, level)                          \
  while (auto* _result = ::tsl::TfCheckOpHelper(val, #val)) \
  LOG(level) << *(_result)

#define TF_CHECK_OK(val) TF_DO_CHECK_OK(val, FATAL)
#define TF_QCHECK_OK(val) TF_DO_CHECK_OK(val, QFATAL)

// DEBUG only version of TF_CHECK_OK.  Compiler still parses 'val' even in opt
// mode.
#ifndef NDEBUG
#define TF_DCHECK_OK(val) TF_CHECK_OK(val)
#else
#define TF_DCHECK_OK(val) \
  while (false && (absl::OkStatus() == (val))) LOG(FATAL)
#endif

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_STATUS_H_
