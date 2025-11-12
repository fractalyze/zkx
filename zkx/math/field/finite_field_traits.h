#ifndef ZKX_MATH_FIELD_FINITE_FIELD_TRAITS_H_
#define ZKX_MATH_FIELD_FINITE_FIELD_TRAITS_H_

namespace zkx::math {

template <typename Config>
class PrimeField;

template <typename Config>
class ExtensionField;

template <typename T>
class FiniteFieldTraits;

template <typename Config>
class FiniteFieldTraits<PrimeField<Config>> {
 public:
  using BasePrimeField = PrimeField<Config>;
};

template <typename Config>
class FiniteFieldTraits<ExtensionField<Config>> {
 public:
  using BasePrimeField = typename Config::BasePrimeField;
};

}  // namespace zkx::math

#endif  // ZKX_MATH_FIELD_FINITE_FIELD_TRAITS_H_
