# Strings

This is taken and modified from
[xla](https://github.com/openxla/xla/tree/8bac4a2/xla/tsl/lib/strings).

```shell
> diff -r /path/to/openxla/xla/xla/tsl/lib/math xla/tsl/lib/math
Only in /Users/chokobole/Workspace/openxla/xla/xla/tsl/lib/strings: BUILD
Only in xla/tsl/lib/strings: BUILD.bazel
Only in xla/tsl/lib/strings: README.md
diff --color -r /Users/chokobole/Workspace/openxla/xla/xla/tsl/lib/strings/proto_serialization.cc xla/tsl/lib/strings/proto_serialization.cc
17c17,18
< #include <cstring>
---
> #include <string.h>
>
20,25c21,22
< #include "absl/memory/memory.h"
< #include "absl/strings/string_view.h"
< #include "xla/tsl/lib/gtl/inlined_vector.h"
< #include "xla/tsl/platform/logging.h"
< #include "xla/tsl/platform/macros.h"
< #include "tsl/platform/hash.h"
---
> #include "google/protobuf/io/coded_stream.h"
> #include "google/protobuf/io/zero_copy_stream_impl_lite.h"
26a24,26
> #include "xla/tsl/platform/hash.h"
> #include "absl/log/check.h"
>
33c33
<   explicit DeterministicSerializer(const protobuf::MessageLite& msg)
---
>   explicit DeterministicSerializer(const google::protobuf::MessageLite& msg)
36c36
<   DeterministicSerializer(const protobuf::MessageLite& msg, size_t size)
---
>   DeterministicSerializer(const google::protobuf::MessageLite& msg, size_t size)
57a58
>
60,61c61,62
< bool SerializeToStringDeterministic(const protobuf::MessageLite& msg,
<                                     string* result) {
---
> bool SerializeToStringDeterministic(const google::protobuf::MessageLite& msg,
>                                     std::string* result) {
64c65
<   *result = string(size, '\0');
---
>   *result = std::string(size, '\0');
69c70
< bool SerializeToBufferDeterministic(const protobuf::MessageLite& msg,
---
> bool SerializeToBufferDeterministic(const google::protobuf::MessageLite& msg,
72,73c73,74
<   protobuf::io::ArrayOutputStream array_stream(buffer, size);
<   protobuf::io::CodedOutputStream output_stream(&array_stream);
---
>   google::protobuf::io::ArrayOutputStream array_stream(buffer, size);
>   google::protobuf::io::CodedOutputStream output_stream(&array_stream);
80,81c81,82
< bool AreSerializedProtosEqual(const protobuf::MessageLite& x,
<                               const protobuf::MessageLite& y) {
---
> bool AreSerializedProtosEqual(const google::protobuf::MessageLite& x,
>                               const google::protobuf::MessageLite& y) {
90,91c91,92
< uint64 DeterministicProtoHash64(const protobuf::MessageLite& proto,
<                                 uint64 seed) {
---
> uint64_t DeterministicProtoHash64(const google::protobuf::MessageLite& proto,
>                                   uint64_t seed) {
96c97
< uint64 DeterministicProtoHash64(const protobuf::MessageLite& proto) {
---
> uint64_t DeterministicProtoHash64(const google::protobuf::MessageLite& proto) {
diff --color -r /Users/chokobole/Workspace/openxla/xla/xla/tsl/lib/strings/proto_serialization.h xla/tsl/lib/strings/proto_serialization.h
18c18,19
< #include "tsl/platform/protobuf.h"
---
> #include <stddef.h>
> #include <stdint.h>
19a21,22
> #include "google/protobuf/message_lite.h"
>
29c32
<                                     string* result);
---
>                                     std::string* result);
44c47
<                                 uint64 seed);
---
>                                 uint64_t seed);
```
