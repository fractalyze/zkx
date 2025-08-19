# GTL

This is taken and modified from
[xla](https://github.com/openxla/xla/tree/8bac4a2/xla/tsl/lib/gtl).

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
44c13,16
<     deps = [":flatset"],
---
>     deps = [
>         ":flatset",
>         "@com_google_absl//absl/log:check",
>     ],
52,54c24,26
<         "//xla/tsl/platform:logging",
<         "//xla/tsl/platform:types",
<         "@tsl//tsl/platform:hash",
---
>         "//xla/tsl/platform:hash",
>         "//zkx/base:logging",
>         "@com_google_absl//absl/log:check",
61,64c33
<     deps = [
<         "//xla/tsl/platform:types",
<         "@com_google_absl//absl/base:prefetch",
<     ],
---
>     deps = ["@com_google_absl//absl/base:prefetch"],
72,74c41,43
<         "//xla/tsl/platform:logging",
<         "//xla/tsl/platform:types",
<         "@tsl//tsl/platform:hash",
---
>         "//xla/tsl/platform:hash",
>         "//zkx/base:logging",
>         "@com_google_absl//absl/log:check",
79,89d47
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
93,94c51,52
<         "//xla/tsl/platform:macros",
<         "//xla/tsl/platform:types",
---
>         "@com_google_absl//absl/base:core_headers",
>         "@com_google_absl//absl/hash",
100a59,62
>     deps = [
>         "@com_google_absl//absl/base:core_headers",
>         "@com_google_absl//absl/hash",
>     ],
105c67
<     srcs = [
---
>     hdrs = [
109d70
<     hdrs = ["map_util.h"],
112,208c73
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
225,229d89
<         "//xla/tsl/platform:macros",
<         "//xla/tsl/platform:test",
<         "//xla/tsl/platform:types",
<         "@com_google_googletest//:gtest_main",
<         "@tsl//tsl/platform:hash",
Only in xla/tsl/lib/gtl: README.md
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/compactptrset.h xla/tsl/lib/gtl/compactptrset.h
18a19
> #include <iterator>
19a21,22
>
> #include "absl/log/check.h"
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/compactptrset_test.cc xla/tsl/lib/gtl/compactptrset_test.cc
18,20c18,19
< #include "xla/tsl/platform/test.h"
< #include "xla/tsl/platform/types.h"
< #include "tsl/platform/hash.h"
---
> #include <algorithm>
> #include <vector>
21a21,22
> #include "gtest/gtest.h"
>
36c37
<   string data = "ABCDEFG";
---
>   std::string data = "ABCDEFG";
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/flatmap.h xla/tsl/lib/gtl/flatmap.h
25a26,27
> #include "absl/log/check.h"
>
27,29c29,30
< #include "xla/tsl/platform/logging.h"
< #include "xla/tsl/platform/types.h"
< #include "tsl/platform/hash.h"
---
> #include "xla/tsl/platform/hash.h"
> #include "zkx/base/logging.h"
136c137
<     iterator(Bucket* b, Bucket* end, uint32 i) : b_(b), end_(end), i_(i) {
---
>     iterator(Bucket* b, Bucket* end, uint32_t i) : b_(b), end_(end), i_(i) {
163c164
<     uint32 i_;
---
>     uint32_t i_;
195c196
<     const_iterator(Bucket* b, Bucket* end, uint32 i) : rep_(b, end, i) {}
---
>     const_iterator(Bucket* b, Bucket* end, uint32_t i) : rep_(b, end, i) {}
324c325
<     uint8 marker[Rep::kWidth];
---
>     uint8_t marker[Rep::kWidth];
336c337
<     Key& key(uint32 i) {
---
>     Key& key(uint32_t i) {
340c341
<     Val& val(uint32 i) {
---
>     Val& val(uint32_t i) {
345c346
<     void InitVal(uint32 i, V&& v) {
---
>     void InitVal(uint32_t i, V&& v) {
348c349
<     void Destroy(uint32 i) {
---
>     void Destroy(uint32_t i) {
352c353
<     void MoveFrom(uint32 i, Bucket* src, uint32 src_index) {
---
>     void MoveFrom(uint32_t i, Bucket* src, uint32_t src_index) {
356c357
<     void CopyFrom(uint32 i, Bucket* src, uint32 src_index) {
---
>     void CopyFrom(uint32_t i, Bucket* src, uint32_t src_index) {
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/flatmap_test.cc xla/tsl/lib/gtl/flatmap_test.cc
25,27c25
< #include "xla/tsl/platform/test.h"
< #include "xla/tsl/platform/types.h"
< #include "tsl/platform/hash.h"
---
> #include "gtest/gtest.h"
33c31
< typedef FlatMap<int64_t, int32> NumMap;
---
> typedef FlatMap<int64_t, int32_t> NumMap;
36c34
< int32 Get(const NumMap& map, int64_t k, int32_t def = -1) {
---
> int32_t Get(const NumMap& map, int64_t k, int32_t def = -1) {
50c48
< typedef std::vector<std::pair<int64_t, int32>> NumMapContents;
---
> typedef std::vector<std::pair<int64_t, int32_t>> NumMapContents;
149,150c147,148
<   FlatMap<int64_t, std::unique_ptr<string>> smap;
<   smap.emplace(1, std::make_unique<string>("hello"));
---
>   FlatMap<int64_t, std::unique_ptr<std::string>> smap;
>   smap.emplace(1, std::make_unique<std::string>("hello"));
347c345
<   typedef std::unordered_map<int64_t, int32> StdNumMap;
---
>   typedef std::unordered_map<int64_t, int32_t> StdNumMap;
594,597c592,595
<   FlatMap<string, string> map;
<   string k1 = "the quick brown fox jumped over the lazy dog";
<   string k2 = k1 + k1;
<   string k3 = k1 + k2;
---
>   FlatMap<std::string, std::string> map;
>   std::string k1 = "the quick brown fox jumped over the lazy dog";
>   std::string k2 = k1 + k1;
>   std::string k3 = k1 + k2;
604c602
<   EXPECT_EQ(string(), map[k3]);
---
>   EXPECT_EQ(std::string(), map[k3]);
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/flatrep.h xla/tsl/lib/gtl/flatrep.h
18a19,20
> #include <stddef.h>
> #include <stdint.h>
24d25
< #include "xla/tsl/platform/types.h"
50,51c51,52
<   static constexpr uint32 kBase = 3;
<   static constexpr uint32 kWidth = (1 << kBase);
---
>   static constexpr uint32_t kBase = 3;
>   static constexpr uint32_t kWidth = (1 << kBase);
105c106
<       for (uint32 i = 0; i < kWidth; i++) {
---
>       for (uint32_t i = 0; i < kWidth; i++) {
137c138
<     uint32 index;
---
>     uint32_t index;
148c149
<     const uint32 marker = Marker(h & 0xff);
---
>     const uint32_t marker = Marker(h & 0xff);
150c151
<     uint32 num_probes = 1;            // Needed for quadratic probing
---
>     uint32_t num_probes = 1;          // Needed for quadratic probing
152c153
<       uint32 bi = index & (kWidth - 1);
---
>       uint32_t bi = index & (kWidth - 1);
154c155
<       const uint32 x = b->marker[bi];
---
>       const uint32_t x = b->marker[bi];
173c174
<     const uint32 marker = Marker(h & 0xff);
---
>     const uint32_t marker = Marker(h & 0xff);
175c176
<     uint32 num_probes = 1;            // Needed for quadratic probing
---
>     uint32_t num_probes = 1;          // Needed for quadratic probing
177c178
<     uint32 di = 0;
---
>     uint32_t di = 0;
179c180
<       uint32 bi = index & (kWidth - 1);
---
>       uint32_t bi = index & (kWidth - 1);
181c182
<       const uint32 x = b->marker[bi];
---
>       const uint32_t x = b->marker[bi];
206c207
<   void Erase(Bucket* b, uint32 i) {
---
>   void Erase(Bucket* b, uint32_t i) {
216c217
<     uint32 bi = index & (kWidth - 1);
---
>     uint32_t bi = index & (kWidth - 1);
250c251
<   uint8 lglen_;       // lg(#buckets)
---
>   uint8_t lglen_;     // lg(#buckets)
261c262
<   static uint32 Marker(uint32 hb) { return hb + (hb < 2 ? 2 : 0); }
---
>   static uint32_t Marker(uint32_t hb) { return hb + (hb < 2 ? 2 : 0); }
293c294,295
<     inline void operator()(Bucket* dst, uint32 dsti, Bucket* src, uint32 srci) {
---
>     inline void operator()(Bucket* dst, uint32_t dsti, Bucket* src,
>                            uint32_t srci) {
300c302,303
<     inline void operator()(Bucket* dst, uint32 dsti, Bucket* src, uint32 srci) {
---
>     inline void operator()(Bucket* dst, uint32_t dsti, Bucket* src,
>                            uint32_t srci) {
310c313
<       for (uint32 i = 0; i < kWidth; i++) {
---
>       for (uint32_t i = 0; i < kWidth; i++) {
323c326
<   void FreshInsert(Bucket* src, uint32 src_index, Copier copier) {
---
>   void FreshInsert(Bucket* src, uint32_t src_index, Copier copier) {
325c328
<     const uint32 marker = Marker(h & 0xff);
---
>     const uint32_t marker = Marker(h & 0xff);
327c330
<     uint32 num_probes = 1;            // Needed for quadratic probing
---
>     uint32_t num_probes = 1;          // Needed for quadratic probing
329c332
<       uint32 bi = index & (kWidth - 1);
---
>       uint32_t bi = index & (kWidth - 1);
331c334
<       const uint32 x = b->marker[bi];
---
>       const uint32_t x = b->marker[bi];
343c346
<   inline size_t NextIndex(size_t i, uint32 num_probes) const {
---
>   inline size_t NextIndex(size_t i, uint32_t num_probes) const {
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/flatset.h xla/tsl/lib/gtl/flatset.h
21,22d20
< #include <functional>
< #include <initializer_list>
25a24,25
> #include "absl/log/check.h"
>
27,29c27,28
< #include "xla/tsl/platform/logging.h"
< #include "xla/tsl/platform/types.h"
< #include "tsl/platform/hash.h"
---
> #include "xla/tsl/platform/hash.h"
> #include "zkx/base/logging.h"
121c120
<     const_iterator(Bucket* b, Bucket* end, uint32 i)
---
>     const_iterator(Bucket* b, Bucket* end, uint32_t i)
146c145
<     uint32 i_;
---
>     uint32_t i_;
260c259
<     uint8 marker[Rep::kWidth];
---
>     uint8_t marker[Rep::kWidth];
269c268
<     Key& key(uint32 i) {
---
>     Key& key(uint32_t i) {
273,274c272,273
<     void Destroy(uint32 i) { storage.key[i].Key::~Key(); }
<     void MoveFrom(uint32 i, Bucket* src, uint32 src_index) {
---
>     void Destroy(uint32_t i) { storage.key[i].Key::~Key(); }
>     void MoveFrom(uint32_t i, Bucket* src, uint32_t src_index) {
277c276
<     void CopyFrom(uint32 i, Bucket* src, uint32 src_index) {
---
>     void CopyFrom(uint32_t i, Bucket* src, uint32_t src_index) {
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/flatset_test.cc xla/tsl/lib/gtl/flatset_test.cc
23,25c23
< #include "xla/tsl/platform/test.h"
< #include "xla/tsl/platform/types.h"
< #include "tsl/platform/hash.h"
---
> #include "gtest/gtest.h"
490,493c488,491
<   FlatSet<string> set;
<   string k1 = "the quick brown fox jumped over the lazy dog";
<   string k2 = k1 + k1;
<   string k3 = k1 + k2;
---
>   FlatSet<std::string> set;
>   std::string k1 = "the quick brown fox jumped over the lazy dog";
>   std::string k2 = k1 + k1;
>   std::string k3 = k1 + k2;
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
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/map_util_test.cc xla/tsl/lib/gtl/map_util_test.cc
22,23c22
< #include "xla/tsl/platform/test.h"
< #include "xla/tsl/platform/types.h"
---
> #include "gtest/gtest.h"
28c27
<   typedef std::map<string, string> Map;
---
>   typedef std::map<std::string, std::string> Map;
42c41
<   typedef std::map<string, string> Map;
---
>   typedef std::map<std::string, std::string> Map;
diff --color -r /path/to/openxla/xla/xla/tsl/lib/gtl/subtle/BUILD xla/tsl/lib/gtl/subtle/BUILD
4,6d3
< load("//xla/tsl:tsl.bzl", "internal_visibility")
< load("//xla/tsl:tsl.default.bzl", "filegroup")
<
8a6
>     default_visibility = ["//visibility:public"],
17,20d14
<     visibility = internal_visibility([
<         "//tensorflow/core/lib/gtl/subtle:__pkg__",
<         "//xla/tsl/lib/gtl:__pkg__",
<     ]),
```
