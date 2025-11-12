#ifndef ZKX_MATH_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_
#define ZKX_MATH_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_

#include "zkx/math/elliptic_curve/short_weierstrass/affine_point.h"
#include "zkx/math/elliptic_curve/short_weierstrass/jacobian_point.h"
#include "zkx/math/elliptic_curve/short_weierstrass/point_xyzz.h"
#include "zkx/math/elliptic_curve/short_weierstrass/sw_curve.h"
#include "zkx/math/field/extension_field.h"
#include "zkx/math/field/prime_field.h"

namespace zkx::math::test {

struct PrimeFieldBaseConfig {
 public:
  constexpr static size_t kModulusBits = 4;
  constexpr static BigInt<1> kModulus = 7;

  constexpr static BigInt<1> kRSquared = UINT64_C(4);
  constexpr static uint64_t kNPrime = UINT64_C(10540996613548315209);

  constexpr static uint32_t kTwoAdicity = 1;

  constexpr static BigInt<1> kTrace = 3;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

struct PrimeFieldStdConfig : public PrimeFieldBaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = PrimeFieldStdConfig;

  constexpr static BigInt<1> kOne = 1;

  constexpr static BigInt<1> kTwoAdicRootOfUnity = 5;
};

struct PrimeFieldConfig : public PrimeFieldBaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = PrimeFieldStdConfig;

  constexpr static BigInt<1> kOne = 2;

  constexpr static BigInt<1> kTwoAdicRootOfUnity = 3;
};

using Fq = PrimeField<PrimeFieldConfig>;
using FqStd = PrimeField<PrimeFieldStdConfig>;
using Fr = PrimeField<PrimeFieldConfig>;
using FrStd = PrimeField<PrimeFieldStdConfig>;

struct Fq2BaseConfig {
  using BaseField = Fq;
  using BasePrimeField = Fq;

  constexpr static uint32_t kDegreeOverBaseField = 2;
  constexpr static BaseField kNonResidue = -1;
};

struct Fq2StdConfig : public Fq2BaseConfig {
  constexpr static bool kUseMontgomery = false;

  using StdConfig = Fq2StdConfig;
};

struct Fq2Config : public Fq2BaseConfig {
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Fq2StdConfig;
};

using Fq2 = ExtensionField<Fq2Config>;
using Fq2Std = ExtensionField<Fq2StdConfig>;

template <typename _BaseField, typename _ScalarField>
class SwCurveBaseConfig {
 public:
  using BaseField = _BaseField;
  using ScalarField = _ScalarField;

  constexpr static BaseField kA = 0;
  constexpr static BaseField kB = 5;
  constexpr static BaseField kX = 5;
  constexpr static BaseField kY = 5;
};

class SwCurveStdConfig : public SwCurveBaseConfig<FqStd, FrStd> {
 public:
  constexpr static bool kUseMontgomery = false;

  using StdConfig = SwCurveStdConfig;
};

class SwCurveConfig : public SwCurveBaseConfig<Fq, Fr> {
 public:
  constexpr static bool kUseMontgomery = true;

  using StdConfig = SwCurveStdConfig;
};

using G1Curve = SwCurve<SwCurveConfig>;
using G1CurveStd = SwCurve<SwCurveStdConfig>;
using AffinePoint = zkx::math::AffinePoint<G1Curve>;
using AffinePointStd = zkx::math::AffinePoint<G1CurveStd>;
using JacobianPoint = zkx::math::JacobianPoint<G1Curve>;
using JacobianPointStd = zkx::math::JacobianPoint<G1CurveStd>;
using PointXyzz = zkx::math::PointXyzz<G1Curve>;
using PointXyzzStd = zkx::math::PointXyzz<G1CurveStd>;

}  // namespace zkx::math::test

#endif  // ZKX_MATH_ELLIPTIC_CURVE_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_
