#ifndef ZKX_BASE_OPENMP_UTIL_H_
#define ZKX_BASE_OPENMP_UTIL_H_

#include <stddef.h>

#include <optional>

#if defined(ZKX_HAS_OPENMP)
#include <omp.h>
#endif  // defined(ZKX_HAS_OPENMP)

#if defined(ZKX_HAS_OPENMP)
#define OMP_PARALLEL_FOR(expr) _Pragma("omp parallel for") for (expr)
#define OMP_PARALLEL_NESTED_FOR(expr) \
  _Pragma("omp parallel for collapse(2)") for (expr)
#else
#define OMP_PARALLEL_FOR(expr) for (expr)
#define OMP_PARALLEL_NESTED_FOR(expr) for (expr)
#endif  // defined(ZKX_HAS_OPENMP)

namespace tachyon::base {

// NOTE(chokobole): This function might return 0. You should handle this case
// carefully. See other examples where it is used.
inline size_t GetSizePerThread(size_t total_size,
                               std::optional<size_t> threshold = std::nullopt) {
#if defined(ZKX_HAS_OPENMP)
  size_t thread_nums = static_cast<size_t>(omp_get_max_threads());
#else
  size_t thread_nums = 1;
#endif
  return (!threshold.has_value() || total_size > threshold.value())
             ? (total_size + thread_nums - 1) / thread_nums
             : total_size;
}

// NOTE(chokobole): This function might return 0. You should handle this case
// carefully. See other examples where it is used.
template <typename Container>
size_t GetNumElementsPerThread(const Container& container,
                               std::optional<size_t> threshold = std::nullopt) {
  return GetSizePerThread(std::size(container));
}

}  // namespace tachyon::base

#endif  // ZKX_BASE_OPENMP_UTIL_H_
