# CORE

This is taken and modified from
[xla](https://github.com/openxla/xla/tree/bb6c362/xla/tsl/lib/core).

```shell
diff -r /path/to/openxla/xla/xla/tsl/lib/core xla/tsl/lib/core
diff --color -r /path/to/openxla/xla/xla/tsl/lib/core/BUILD xla/tsl/lib/core/BUILD
7,13c7
< load("//xla/tsl:tsl.bzl", "internal_visibility")
< load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
< load("//xla/tsl/platform:build_config.bzl", "tsl_cc_test")
< load(
<     "//xla/tsl/platform:rules_cc.bzl",
<     "cc_library",
< )
---
> load("@rules_cc//cc:defs.bzl", "cc_library")
24,36d17
< filegroup(
<     name = "legacy_lib_core_status_test_util_header",
<     srcs = [
<         "status_test_util.h",
<     ],
<     compatible_with = get_compatible_with_portable(),
<     visibility = internal_visibility([
<         "//tensorflow/core:__pkg__",
<         "//xla/tsl/lib/core:__pkg__",
<         "//tensorflow/core/lib/core:__pkg__",
<     ]),
< )
<
38,92d18
<     name = "status_test_util",
<     testonly = 1,
<     hdrs = ["status_test_util.h"],
<     compatible_with = get_compatible_with_portable(),
<     deps = [
<         "//xla/tsl/platform:status_matchers",
<         "//xla/tsl/platform:test",
<         "@com_google_absl//absl/status:status_matchers",
<     ],
< )
<
< filegroup(
<     name = "mobile_srcs_only_runtime",
<     srcs = [
<         "bitmap.h",
<         "bits.h",
<     ],
<     compatible_with = get_compatible_with_portable(),
<     visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
< )
<
< filegroup(
<     name = "legacy_lib_core_all_headers",
<     srcs = [
<         "bitmap.h",
<         "bits.h",
<         "status_test_util.h",
<     ],
<     compatible_with = get_compatible_with_portable(),
<     visibility = internal_visibility([
<         "//tensorflow/core:__pkg__",
<         "//tensorflow/core/lib/core:__pkg__",
<     ]),
< )
<
< filegroup(
<     name = "legacy_lib_core_all_tests",
<     srcs = [
<         "bitmap_test.cc",
<     ],
<     compatible_with = get_compatible_with_portable(),
<     visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
< )
<
< filegroup(
<     name = "legacy_lib_core_headers",
<     srcs = [
<         "bitmap.h",
<         "bits.h",
<     ],
<     compatible_with = get_compatible_with_portable(),
<     visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
< )
<
< cc_library(
96d21
<     compatible_with = get_compatible_with_portable(),
98c23
<         "//xla/tsl/platform:logging",
---
>         "@com_google_absl//absl/log:check",
102,134d26
< )
<
< tsl_cc_test(
<     name = "bitmap_test",
<     size = "small",
<     srcs = ["bitmap_test.cc"],
<     deps = [
<         ":bitmap",
<         "//xla/tsl/lib/random:philox",
<         "//xla/tsl/lib/random:philox_random",
<         "//xla/tsl/platform:test",
<         "@com_google_googletest//:gtest_main",
<     ],
< )
<
< cc_library(
<     name = "bits",
<     hdrs = ["bits.h"],
<     deps = [
<         "//xla/tsl/platform:logging",
<         "@com_google_absl//absl/numeric:bits",
<     ],
< )
<
< tsl_cc_test(
<     name = "bits_test",
<     size = "small",
<     srcs = ["bits_test.cc"],
<     deps = [
<         ":bits",
<         "//xla/tsl/platform:test",
<         "@com_google_googletest//:gtest_main",
<     ],
Only in xla/tsl/lib/core: README.md
diff --color -r /path/to/openxla/xla/xla/tsl/lib/core/bitmap.h xla/tsl/lib/core/bitmap.h
24c24
< #include "xla/tsl/platform/logging.h"
---
> #include "absl/log/check.h"
Only in /path/to/openxla/xla/xla/tsl/lib/core: bitmap_test.cc
Only in /path/to/openxla/xla/xla/tsl/lib/core: bits.h
Only in /path/to/openxla/xla/xla/tsl/lib/core: bits_test.cc
Only in /path/to/openxla/xla/xla/tsl/lib/core: status_test_util.h
```
