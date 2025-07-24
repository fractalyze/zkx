#ifndef ZKX_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_
#define ZKX_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_

#include "zkx/math/base/extension_field.h"
#include "zkx/math/base/prime_field.h"
#include "zkx/math/elliptic_curves/short_weierstrass/affine_point.h"
#include "zkx/math/elliptic_curves/short_weierstrass/jacobian_point.h"
#include "zkx/math/elliptic_curves/short_weierstrass/point_xyzz.h"
#include "zkx/math/elliptic_curves/short_weierstrass/sw_curve.h"

namespace zkx::math::test {

struct PrimeFieldConfig {
 public:
  constexpr static size_t kModulusBits = 4;
  constexpr static BigInt<1> kModulus = 7;
  constexpr static BigInt<1> kOne = 2;

  constexpr static BigInt<1> kRSquared = UINT64_C(4);
  constexpr static uint64_t kNPrime = UINT64_C(10540996613548315209);

  constexpr static uint32_t kTwoAdicity = 1;

  constexpr static BigInt<1> kTrace = 3;

  constexpr static bool kHasTwoAdicRootOfUnity = true;
  constexpr static BigInt<1> kTwoAdicRootOfUnity = 3;

  constexpr static bool kHasLargeSubgroupRootOfUnity = false;
};

using Fq = PrimeField<PrimeFieldConfig>;
using Fr = PrimeField<PrimeFieldConfig>;

struct Fq2Config {
  using BaseField = Fq;
  using BasePrimeField = Fq;

  constexpr static uint32_t kDegreeOverBaseField = 2;
  constexpr static BaseField kNonResidue = -1;
};

using Fq2 = ExtensionField<Fq2Config>;

template <typename _BaseField, typename _ScalarField>
class SwCurveConfig {
 public:
  using BaseField = _BaseField;
  using ScalarField = _ScalarField;

  constexpr static BaseField kA = 0;
  constexpr static BaseField kB = 5;
  constexpr static BaseField kX = 5;
  constexpr static BaseField kY = 5;
};

using G1Curve = SwCurve<SwCurveConfig<Fq, Fr>>;
using AffinePoint = zkx::math::AffinePoint<G1Curve>;
using JacobianPoint = zkx::math::JacobianPoint<G1Curve>;
using PointXyzz = zkx::math::PointXyzz<G1Curve>;

}  // namespace zkx::math::test

#endif  // ZKX_MATH_ELLIPTIC_CURVES_SHORT_WEIERSTRASS_TEST_SW_CURVE_CONFIG_H_
