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

#ifndef XLA_TSL_PLATFORM_ERRORS_H_
#define XLA_TSL_PLATFORM_ERRORS_H_

#include "absl/base/optimization.h"
#include "absl/status/status.h"

// For propagating errors when calling a function.
#define TF_RETURN_IF_ERROR(...)              \
  do {                                       \
    absl::Status _status = (__VA_ARGS__);    \
    if (ABSL_PREDICT_FALSE(!_status.ok())) { \
      return _status;                        \
    }                                        \
  } while (0)

#endif  // XLA_TSL_PLATFORM_ERRORS_H_
