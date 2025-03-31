# GTL

This is taken and modified from [xla](https://github.com/openxla/xla/tree/8bac4a2/xla/tsl/lib/gtl).

```shell
diff -r /path/to/openxla/xla/xla/tsl/lib/gtl xla/tsl/lib/gtl
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/BUILD xla/tsl/lib/gtl/BUILD
1,10c1,2
< load("//xla/tsl:tsl.bzl", "internal_visibility")
< load("//xla/tsl:tsl.default.bzl", "filegroup")
< load(
<     "//xla/tsl/platform:build_config.bzl",
<     "tsl_cc_test",
< )
< load(
<     "//xla/tsl/platform:rules_cc.bzl",
<     "cc_library",
< )
---
> load("@rules_cc//cc:defs.bzl", "cc_library")
> load("//bazel:zkx_cc.bzl", "zkx_cc_unittest")
14,35c6
<     default_visibility = internal_visibility([
<         # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
<         "//tensorflow/core:__pkg__",
<         # tensorflow/core/lib/strings:proto_serialization uses on gtl:inlined_vector
<         "//tensorflow/core/lib/strings:__pkg__",
<         "//xla/tsl/lib/strings:__pkg__",
<         # tensorflow/core/framework uses map_util, and flatmap
<         "//tensorflow/core/framework:__pkg__",
<         "//xla/tsl/framework:__pkg__",
<         "//xla/tsl/platform/cloud:__pkg__",
<         # tensorflow/core/util uses inlined_vector
<         "//tensorflow/core/util:__pkg__",
<         # tensorflow/core/tfrt/utils uses inlined_vector
<         "//tensorflow/core/tfrt/utils:__pkg__",
<         # tensorflow/examples/custom_ops_doc/simple_hash_table uses map_util
<         "//tensorflow/examples/custom_ops_doc/simple_hash_table:__pkg__",
<         "//xla:__subpackages__",
<         "//tensorflow/core/lib/gtl:__subpackages__",
<         "//xla/tsl/distributed_runtime/rpc:__pkg__",
<         "//xla/tsl/profiler/utils:__pkg__",
<         "//tensorflow/core/profiler/convert:__pkg__",
<     ]),
---
>     default_visibility = ["//visibility:public"],
39,40d9
< # Todo(bmzhao): Remaining targets to add to this BUILD file are: all tests.
<
42,89d10
<     name = "compactptrset",
<     hdrs = ["compactptrset.h"],
<     deps = [":flatset"],
< )
<
< cc_library(
<     name = "flatmap",
<     hdrs = ["flatmap.h"],
<     deps = [
<         ":flatrep",
<         "//xla/tsl/platform:logging",
<         "//xla/tsl/platform:types",
<         "@tsl//tsl/platform:hash",
<     ],
< )
<
< cc_library(
<     name = "flatrep",
<     hdrs = ["flatrep.h"],
<     deps = [
<         "//xla/tsl/platform:types",
<         "@com_google_absl//absl/base:prefetch",
<     ],
< )
<
< cc_library(
<     name = "flatset",
<     hdrs = ["flatset.h"],
<     deps = [
<         ":flatrep",
<         "//xla/tsl/platform:logging",
<         "//xla/tsl/platform:types",
<         "@tsl//tsl/platform:hash",
<     ],
< )
<
< cc_library(
<     name = "inlined_vector",
<     hdrs = ["inlined_vector.h"],
<     deps = [
<         "//xla/tsl/platform:macros",
<         "//xla/tsl/platform:types",
<         "@com_google_absl//absl/base:core_headers",
<         "@com_google_absl//absl/container:inlined_vector",
<     ],
< )
<
< cc_library(
93,94c14,15
<         "//xla/tsl/platform:macros",
<         "//xla/tsl/platform:types",
---
>         "@com_google_absl//absl/base:core_headers",
>         "@com_google_absl//absl/hash",
101,107c22,24
< )
<
< cc_library(
<     name = "map_util",
<     srcs = [
<         "map_util.h",
<         "//xla/tsl/lib/gtl/subtle:map_traits",
---
>     deps = [
>         "@com_google_absl//absl/base:core_headers",
>         "@com_google_absl//absl/hash",
109d25
<     hdrs = ["map_util.h"],
112,208c28
< filegroup(
<     name = "legacy_lib_gtl_headers",
<     srcs = [
<         "compactptrset.h",
<         "flatmap.h",
<         "flatset.h",
<         "inlined_vector.h",
<         "iterator_range.h",
<     ],
<     visibility = internal_visibility([
<         "//tensorflow/core:__pkg__",
<         "//tensorflow/core/lib/gtl:__pkg__",
<     ]),
< )
<
< filegroup(
<     name = "legacy_lib_internal_public_gtl_headers",
<     srcs = [
<         "int_type.h",
<         "map_util.h",
<     ],
<     visibility = internal_visibility([
<         "//tensorflow/core:__pkg__",
<         "//tensorflow/core/lib/gtl:__pkg__",
<     ]),
< )
<
< filegroup(
<     name = "legacy_lib_test_internal_headers",
<     srcs = [
<     ],
<     visibility = internal_visibility([
<         "//tensorflow/core:__pkg__",
<         "//tensorflow/core/lib/gtl:__pkg__",
<     ]),
< )
<
< filegroup(
<     name = "legacy_android_gif_internal_headers",
<     srcs = [
<     ],
<     visibility = internal_visibility([
<         "//tensorflow/core:__pkg__",
<         "//tensorflow/core/lib/gtl:__pkg__",
<     ]),
< )
<
< # Export source files needed for mobile builds, which do not use granular targets.
< filegroup(
<     name = "mobile_srcs_no_runtime",
<     srcs = [
<         "flatmap.h",
<         "flatrep.h",
<         "inlined_vector.h",
<     ],
<     visibility = internal_visibility([
<         "//tensorflow/core:__pkg__",
<         "//tensorflow/core/lib/gtl:__pkg__",
<         "@tsl//tsl:__subpackages__",
<     ]),
< )
<
< filegroup(
<     name = "mobile_srcs_only_runtime",
<     srcs = [
<         "flatset.h",
<         "int_type.h",
<         "iterator_range.h",
<         "map_util.h",
<         "//xla/tsl/lib/gtl/subtle:map_traits",
<     ],
<     visibility = internal_visibility([
<         "//tensorflow/core:__pkg__",
<         "//tensorflow/core/lib/gtl:__pkg__",
<     ]),
< )
<
< filegroup(
<     name = "legacy_lib_gtl_all_headers",
<     srcs = [
<         "compactptrset.h",
<         "flatmap.h",
<         "flatrep.h",
<         "flatset.h",
<         "inlined_vector.h",
<         "int_type.h",
<         "iterator_range.h",
<         "map_util.h",
<         "//xla/tsl/lib/gtl/subtle:map_traits",
<     ],
<     visibility = internal_visibility([
<         "//tensorflow/core:__pkg__",
<         "//tensorflow/core/lib/gtl:__pkg__",
<     ]),
< )
<
< tsl_cc_test(
---
> zkx_cc_unittest(
211,213d30
<         "compactptrset_test.cc",
<         "flatmap_test.cc",
<         "flatset_test.cc",
216d32
<         "map_util_test.cc",
219,221d34
<         ":compactptrset",
<         ":flatmap",
<         ":flatset",
224,229d36
<         ":map_util",
<         "//xla/tsl/platform:macros",
<         "//xla/tsl/platform:test",
<         "//xla/tsl/platform:types",
<         "@com_google_googletest//:gtest_main",
<         "@tsl//tsl/platform:hash",
Only in xla/tsl/lib/gtl: README.md
Only in /path/to/openxla/xla/xla/tsl/lib/gtl: compactptrset.h
Only in /path/to/openxla/xla/xla/tsl/lib/gtl: compactptrset_test.cc
Only in /path/to/openxla/xla/xla/tsl/lib/gtl: flatmap.h
Only in /path/to/openxla/xla/xla/tsl/lib/gtl: flatmap_test.cc
Only in /path/to/openxla/xla/xla/tsl/lib/gtl: flatrep.h
Only in /path/to/openxla/xla/xla/tsl/lib/gtl: flatset.h
Only in /path/to/openxla/xla/xla/tsl/lib/gtl: flatset_test.cc
Only in /path/to/openxla/xla/xla/tsl/lib/gtl: inlined_vector.h
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/int_type.h xla/tsl/lib/gtl/int_type.h
162,163c162,163
< #include "xla/tsl/platform/macros.h"
< #include "xla/tsl/platform/types.h"
---
> #include "absl/base/attributes.h"
> #include "absl/hash/hash.h"
254,261c254,261
< #define INT_TYPE_ASSIGNMENT_OP(op)                   \
<   ThisType& operator op(const ThisType& arg_value) { \
<     value_ op arg_value.value();                     \
<     return *this;                                    \
<   }                                                  \
<   ThisType& operator op(ValueType arg_value) {       \
<     value_ op arg_value;                             \
<     return *this;                                    \
---
> #define INT_TYPE_ASSIGNMENT_OP(op)                    \
>   ThisType& operator op(const ThisType & arg_value) { \
>     value_ op arg_value.value();                      \
>     return *this;                                     \
>   }                                                   \
>   ThisType& operator op(ValueType arg_value) {        \
>     value_ op arg_value;                              \
>     return *this;                                     \
282c282
< } TF_PACKED;
---
> } ABSL_ATTRIBUTE_PACKED;
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/int_type_test.cc xla/tsl/lib/gtl/int_type_test.cc
23,24c23
< #include "xla/tsl/platform/test.h"
< #include "xla/tsl/platform/types.h"
---
> #include "gtest/gtest.h"
28,32c27,31
< TSL_LIB_GTL_DEFINE_INT_TYPE(Int8_IT, int8);
< TSL_LIB_GTL_DEFINE_INT_TYPE(UInt8_IT, uint8);
< TSL_LIB_GTL_DEFINE_INT_TYPE(Int16_IT, int16);
< TSL_LIB_GTL_DEFINE_INT_TYPE(UInt16_IT, uint16);
< TSL_LIB_GTL_DEFINE_INT_TYPE(Int32_IT, int32);
---
> TSL_LIB_GTL_DEFINE_INT_TYPE(Int8_IT, int8_t);
> TSL_LIB_GTL_DEFINE_INT_TYPE(UInt8_IT, uint8_t);
> TSL_LIB_GTL_DEFINE_INT_TYPE(Int16_IT, int16_t);
> TSL_LIB_GTL_DEFINE_INT_TYPE(UInt16_IT, uint16_t);
> TSL_LIB_GTL_DEFINE_INT_TYPE(Int32_IT, int32_t);
34,35c33,34
< TSL_LIB_GTL_DEFINE_INT_TYPE(UInt32_IT, uint32);
< TSL_LIB_GTL_DEFINE_INT_TYPE(UInt64_IT, uint64);
---
> TSL_LIB_GTL_DEFINE_INT_TYPE(UInt32_IT, uint32_t);
> TSL_LIB_GTL_DEFINE_INT_TYPE(UInt64_IT, uint64_t);
255,258c254,257
<   EXPECT_EQ(static_cast<int8>(i), int_type.template value<int8>());
<   EXPECT_EQ(static_cast<int16>(i), int_type.template value<int16>());
<   EXPECT_EQ(static_cast<int32>(i), int_type.template value<int32>());
<   EXPECT_EQ(static_cast<uint32>(i), int_type.template value<uint32>());
---
>   EXPECT_EQ(static_cast<int8_t>(i), int_type.template value<int8_t>());
>   EXPECT_EQ(static_cast<int16_t>(i), int_type.template value<int16_t>());
>   EXPECT_EQ(static_cast<int32_t>(i), int_type.template value<int32_t>());
>   EXPECT_EQ(static_cast<uint32_t>(i), int_type.template value<uint32_t>());
260c259
<   EXPECT_EQ(static_cast<uint64>(i), int_type.template value<uint64>());
---
>   EXPECT_EQ(static_cast<uint64_t>(i), int_type.template value<uint64_t>());
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/iterator_range_test.cc xla/tsl/lib/gtl/iterator_range_test.cc
20,22c20
< #include "xla/tsl/platform/macros.h"
< #include "xla/tsl/platform/test.h"
< #include "xla/tsl/platform/types.h"
---
> #include "gtest/gtest.h"
57c55
<     ASSERT_LT(index, TF_ARRAYSIZE(v));
---
>     ASSERT_LT(index, std::size(v));
69c67
<     ASSERT_LT(index, TF_ARRAYSIZE(v));
---
>     ASSERT_LT(index, std::size(v));
Only in /path/to/openxla/xla/xla/tsl/lib/gtl: map_util.h
Only in /path/to/openxla/xla/xla/tsl/lib/gtl: map_util_test.cc
Only in /path/to/openxla/xla/xla/tsl/lib/gtl: subtle
```
