/* Copyright 2021 The OpenXLA Authors.
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

#ifndef ZKX_SERVICE_CUSTOM_CALL_STATUS_H_
#define ZKX_SERVICE_CUSTOM_CALL_STATUS_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ABI-stable public interfaces for ZkxCustomCallStatus.

// Represents the result of a CustomCall: success or failure, with an associated
// error message in the failure case.
typedef struct ZkxCustomCallStatus_ ZkxCustomCallStatus;

// Set the ZkxCustomCallStatus to a success state. This is the default state.
void ZkxCustomCallStatusSetSuccess(ZkxCustomCallStatus* status);

// Set the ZkxCustomCallStatus to a failure state with the given error message.
// Does not take ownership of the supplied message string; instead copies the
// first `message_len` bytes, or up to the null terminator, whichever comes
// first.
void ZkxCustomCallStatusSetFailure(ZkxCustomCallStatus* status,
                                   const char* message, size_t message_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ZKX_SERVICE_CUSTOM_CALL_STATUS_H_
