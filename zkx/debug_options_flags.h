/* Copyright 2017 The OpenXLA Authors.

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

#ifndef ZKX_DEBUG_OPTIONS_FLAGS_H_
#define ZKX_DEBUG_OPTIONS_FLAGS_H_

#include <vector>

#include "xla/tsl/util/command_line_flags.h"
#include "zkx/zkx.pb.h"

namespace zkx {

// Construct flags which write to the debug_options proto when parsed. Existing
// contents of debug_options is used as the default. Can be called multiple
// times.
void MakeDebugOptionsFlags(std::vector<tsl::Flag>* flag_list,
                           DebugOptions* debug_options);

// Appends flag definitions for debug options to flag_list. Existing
// contents of debug_options is used as the default. If debug_options is null,
// uses global defaults. Modifies global state on first call.
void AppendDebugOptionsFlags(std::vector<tsl::Flag>* flag_list,
                             DebugOptions* debug_options = nullptr);

// Fetches a DebugOptions proto message from flags provided to the program.
// Flags must be registered with the flags parser using AppendDebugOptionsFlags
// first.
DebugOptions GetDebugOptionsFromFlags();

// Gets a DebugOptions proto that reflects the defaults as if no flags were set.
DebugOptions DefaultDebugOptionsIgnoringFlags();

}  // namespace zkx

#endif  // ZKX_DEBUG_OPTIONS_FLAGS_H_
