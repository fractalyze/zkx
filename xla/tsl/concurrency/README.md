# Concurrency

This is taken and modified from [xla](https://github.com/openxla/xla/tree/8bac4a2/xla/tsl/concurrency).

```shell
> diff -r /path/to/openxla/xla/xla/tsl/concurrency xla/tsl/concurrency
Only in /path/to/openxla/xla/xla/tsl/concurrency: BUILD
Only in xla/tsl/concurrency: BUILD.bazel
Only in xla/tsl/concurrency: README.md
diff --color -r /path/to/openxla/xla/xla/tsl/concurrency/async_value.cc xla/tsl/concurrency/async_value.cc
26a27
> #include "absl/log/log.h"
28a30
>
31d32
< #include "xla/tsl/platform/logging.h"
diff --color -r /path/to/openxla/xla/xla/tsl/concurrency/async_value.h xla/tsl/concurrency/async_value.h
30a31,32
> #include "absl/log/check.h"
> #include "absl/log/log.h"
32a35
>
35d37
< #include "xla/tsl/platform/logging.h"
diff --color -r /path/to/openxla/xla/xla/tsl/concurrency/async_value_ptr_test.cc xla/tsl/concurrency/async_value_ptr_test.cc
25a26,27
> #include "gtest/gtest.h"
>
28,29d29
< #include "xla/tsl/platform/test.h"
< #include "xla/tsl/platform/test_benchmark.h"
619,651d618
<
< //===----------------------------------------------------------------------===//
< // Performance benchmarks below
< //===----------------------------------------------------------------------===//
<
< struct InlineExecutor : public AsyncValue::Executor {
<   void Execute(Task task) final { task(); }
< };
<
< static void BM_MapIntToFloat(benchmark::State& state) {
<   auto ref = MakeAvailableAsyncValueRef<int32_t>(42);
<   auto ptr = ref.AsPtr();
<
<   for (auto _ : state) {
<     auto mapped = ptr.Map([](int32_t value) -> float { return value; });
<     benchmark::DoNotOptimize(mapped);
<   }
< }
<
< static void BM_MapIntToFloatOnExecutor(benchmark::State& state) {
<   auto ref = MakeAvailableAsyncValueRef<int32_t>(42);
<   auto ptr = ref.AsPtr();
<
<   InlineExecutor executor;
<   for (auto _ : state) {
<     auto mapped =
<         ptr.Map(executor, [](int32_t value) -> float { return value; });
<     benchmark::DoNotOptimize(mapped);
<   }
< }
<
< BENCHMARK(BM_MapIntToFloat);
< BENCHMARK(BM_MapIntToFloatOnExecutor);
diff --color -r /path/to/openxla/xla/xla/tsl/concurrency/async_value_ref.cc xla/tsl/concurrency/async_value_ref.cc
19a20
> #include "absl/log/log.h"
21a23
>
24d25
< #include "xla/tsl/platform/logging.h"
38,39c39,40
< RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(absl::string_view message) {
<   // Converting to `absl::string_view` because implicit conversion is not
---
> RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(std::string_view message) {
>   // Converting to `std::string_view` because implicit conversion is not
41c42
<   absl::string_view message_view(message.data(), message.size());
---
>   std::string_view message_view(message.data(), message.size());
diff --color -r /path/to/openxla/xla/xla/tsl/concurrency/async_value_ref.h xla/tsl/concurrency/async_value_ref.h
32a33
> #include "absl/log/log.h"
37a39
>
40d41
< #include "xla/tsl/platform/logging.h"
350,351c351,352
<   void SetError(absl::string_view message) const {
<     // Converting to `absl::string_view` because implicit conversion is not
---
>   void SetError(std::string_view message) const {
>     // Converting to `std::string_view` because implicit conversion is not
353c354
<     absl::string_view message_view(message.data(), message.size());
---
>     std::string_view message_view(message.data(), message.size());
diff --color -r /path/to/openxla/xla/xla/tsl/concurrency/async_value_ref_test.cc xla/tsl/concurrency/async_value_ref_test.cc
29a30,31
> #include "gtest/gtest.h"
>
32,33d33
< #include "xla/tsl/platform/test.h"
< #include "xla/tsl/platform/test_benchmark.h"
942,992d941
<
< //===----------------------------------------------------------------------===//
< // Performance benchmarks below
< //===----------------------------------------------------------------------===//
<
< template <size_t size>
< static void BM_MakeConstructed(benchmark::State& state) {
<   for (auto _ : state) {
<     auto ref = MakeConstructedAsyncValueRef<std::array<char, size>>();
<     benchmark::DoNotOptimize(ref);
<   }
< }
<
< BENCHMARK(BM_MakeConstructed<1>);
< BENCHMARK(BM_MakeConstructed<4>);
< BENCHMARK(BM_MakeConstructed<8>);
< BENCHMARK(BM_MakeConstructed<16>);
< BENCHMARK(BM_MakeConstructed<32>);
< BENCHMARK(BM_MakeConstructed<64>);
< BENCHMARK(BM_MakeConstructed<128>);
< BENCHMARK(BM_MakeConstructed<256>);
<
< static void BM_CountDownSuccess(benchmark::State& state) {
<   size_t n = state.range(0);
<
<   for (auto _ : state) {
<     auto ref = MakeConstructedAsyncValueRef<int32_t>(42);
<     CountDownAsyncValueRef<int32_t> count_down_ref(ref, n);
<     for (size_t i = 0; i < n; ++i) {
<       count_down_ref.CountDown();
<     }
<   }
< }
<
< BENCHMARK(BM_CountDownSuccess)->Arg(4)->Arg(8)->Arg(16)->Arg(32);
<
< static void BM_CountDownError(benchmark::State& state) {
<   size_t n = state.range(0);
<
<   absl::Status error = absl::InternalError("error");
<
<   for (auto _ : state) {
<     auto ref = MakeConstructedAsyncValueRef<int32_t>(42);
<     CountDownAsyncValueRef<int32_t> count_down_ref(ref, n);
<     for (size_t i = 0; i < n; ++i) {
<       count_down_ref.CountDown(error);
<     }
<   }
< }
<
< BENCHMARK(BM_CountDownError)->Arg(4)->Arg(8)->Arg(16)->Arg(32);
diff --color -r /path/to/openxla/xla/xla/tsl/concurrency/async_value_test.cc xla/tsl/concurrency/async_value_test.cc
22a23,24
> #include "gtest/gtest.h"
>
24d25
< #include "xla/tsl/platform/test.h"
diff --color -r /path/to/openxla/xla/xla/tsl/concurrency/concurrent_vector.h xla/tsl/concurrency/concurrent_vector.h
26a27
> #include "absl/log/check.h"
29d29
< #include "xla/tsl/platform/logging.h"
diff --color -r /path/to/openxla/xla/xla/tsl/concurrency/concurrent_vector_test.cc xla/tsl/concurrency/concurrent_vector_test.cc
21,23c21,23
< #include "xla/tsl/platform/env.h"
< #include "xla/tsl/platform/test.h"
< #include "xla/tsl/platform/threadpool.h"
---
> #include "gtest/gtest.h"
>
> #include "xla/tsl/platform/thread_pool.h"
```

The implementations are the same, but there are some differences in `BUILD.bazel`.
