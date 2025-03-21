# Math

This is taken and modified from [xla](https://github.com/openxla/xla/tree/8bac4a2/xla/tsl/lib/math).

```shell
> diff -r /path/to/openxla/xla/xla/tsl/lib/math xla/tsl/lib/math
Only in /path/to/openxla/xla/xla/tsl/lib/math: BUILD
Only in xla/tsl/lib/math: BUILD.bazel
Only in xla/tsl/lib/math: README.md
diff --color -r /path/to/openxla/xla/xla/tsl/lib/math/math_util_test.cc xla/tsl/lib/math/math_util_test.cc
22,25c22,23
< #include "xla/tsl/platform/logging.h"
< #include "xla/tsl/platform/test.h"
< #include "xla/tsl/platform/test_benchmark.h"
< #include "xla/tsl/platform/types.h"
---
> #include "absl/base/attributes.h"
> #include "gtest/gtest.h"
63c61
< void TestCeilOfRatioUnsigned(uint64 kMax) {
---
> void TestCeilOfRatioUnsigned(uint64_t kMax) {
65c63
<   const uint64 kTestData[kNumTests][kNumTestArguments] = {
---
>   const uint64_t kTestData[kNumTests][kNumTestArguments] = {
86c84
<   TestCeilOfRatio<UnsignedIntegralType, uint64>(kTestData, kNumTests);
---
>   TestCeilOfRatio<UnsignedIntegralType, uint64_t>(kTestData, kNumTests);
184c182
< void TestThatCeilOfRatioDenomMinusOneIsIncorrect() {
---
> ABSL_ATTRIBUTE_UNUSED void TestThatCeilOfRatioDenomMinusOneIsIncorrect() {
192,199c190,197
<   TestCeilOfRatioUnsigned<uint8>(kuint8max);
<   TestCeilOfRatioUnsigned<uint16>(kuint16max);
<   TestCeilOfRatioUnsigned<uint32>(kuint32max);
<   TestCeilOfRatioUnsigned<uint64>(kuint64max);
<   TestCeilOfRatioSigned<int8>(kint8min, kint8max);
<   TestCeilOfRatioSigned<int16>(kint16min, kint16max);
<   TestCeilOfRatioSigned<int32>(kint32min, kint32max);
<   TestCeilOfRatioSigned<int64_t>(kint64min, kint64max);
---
>   TestCeilOfRatioUnsigned<uint8_t>(UINT8_MAX);
>   TestCeilOfRatioUnsigned<uint16_t>(UINT16_MAX);
>   TestCeilOfRatioUnsigned<uint32_t>(UINT32_MAX);
>   TestCeilOfRatioUnsigned<uint64_t>(UINT64_MAX);
>   TestCeilOfRatioSigned<int8_t>(INT8_MIN, INT8_MAX);
>   TestCeilOfRatioSigned<int16_t>(INT16_MIN, INT16_MAX);
>   TestCeilOfRatioSigned<int32_t>(INT32_MIN, INT32_MAX);
>   TestCeilOfRatioSigned<int64_t>(INT64_MIN, INT64_MAX);
223,226c221,224
<     EXPECT_EQ(tc.gcd, tsl::MathUtil::GCD<uint32>(tc.x, tc.y));
<     EXPECT_EQ(tc.gcd, tsl::MathUtil::GCD<uint32>(tc.y, tc.x));
<     EXPECT_EQ(tc.gcd, tsl::MathUtil::GCD<uint64>(tc.x, tc.y));
<     EXPECT_EQ(tc.gcd, tsl::MathUtil::GCD<uint64>(tc.y, tc.x));
---
>     EXPECT_EQ(tc.gcd, tsl::MathUtil::GCD<uint32_t>(tc.x, tc.y));
>     EXPECT_EQ(tc.gcd, tsl::MathUtil::GCD<uint32_t>(tc.y, tc.x));
>     EXPECT_EQ(tc.gcd, tsl::MathUtil::GCD<uint64_t>(tc.x, tc.y));
>     EXPECT_EQ(tc.gcd, tsl::MathUtil::GCD<uint64_t>(tc.y, tc.x));
229c227
<   const uint64 biggish_prime = 1666666667;
---
>   const uint64_t biggish_prime = 1666666667;
231c229
<             tsl::MathUtil::GCD<uint64>(biggish_prime * 3, biggish_prime * 4));
---
>             tsl::MathUtil::GCD<uint64_t>(biggish_prime * 3, biggish_prime * 4));
```
