load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load(
    "//bazel:zkx.bzl",
    "if_has_exception",
    "if_has_openmp",
    "if_has_rtti",
)

def zkx_safe_code():
    return [
        "-Wall",
        "-Werror",
        # NOTE(chokobole): See the whole list of warnings in https://github.com/openxla/xla/tree/8bac4a2/warnings.bazelrc.
        # TODO(chokobole): Remove this warning once resolved. This is required to compile async_value.h.
        # ./xla/tsl/concurrency/async_value.h:722:19: warning: offset of on non-standard-layout type 'ConcreteAsyncValue<C>' [-Winvalid-offsetof]
        # 722 |     static_assert(offsetof(ConcreteAsyncValue<T>, data_store_.data_) ==
        "-Wno-invalid-offsetof",
    ]

def zkx_warnings(safe_code):
    warnings = []
    if safe_code:
        warnings.extend(zkx_safe_code())
    return warnings

def zkx_exceptions(force_exceptions):
    return if_has_exception(["-fexceptions"], (["-fexceptions"] if force_exceptions else ["-fno-exceptions"]))

def zkx_rtti(force_rtti):
    return if_has_rtti(["-frtti"], (["-frtti"] if force_rtti else ["-fno-rtti"]))

def zkx_copts(safe_code = True):
    return zkx_warnings(safe_code)

def zkx_cxxopts(safe_code = True, force_exceptions = False, force_rtti = False):
    return zkx_copts(safe_code) + zkx_exceptions(force_exceptions) + zkx_rtti(force_rtti)

def zkx_openmp_defines():
    return if_has_openmp(["ZKX_HAS_OPENMP"])

def zkx_defines():
    return zkx_openmp_defines()

def zkx_local_defines():
    return []

def zkx_openmp_linkopts():
    return select({
        "@zkx//:zkx_has_openmp_on_macos": ["-Xclang -fopenmp"],
        "@zkx//:zkx_has_openmp": ["-fopenmp"],
        "@zkx//:zkx_has_intel_openmp": ["-liomp5"],
        "//conditions:default": [],
    })

def zkx_linkopts():
    return zkx_openmp_linkopts()

def zkx_openmp_num_threads_env(n):
    return if_has_openmp({
        "OMP_NUM_THREADS": "{}".format(n),
    }, {})

def zkx_cc_library(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        linkopts = [],
        alwayslink = True,
        safe_code = True,
        force_exceptions = False,
        force_rtti = False,
        **kwargs):
    cc_library(
        name = name,
        copts = copts + zkx_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions, force_rtti = force_rtti),
        defines = defines + zkx_defines(),
        local_defines = local_defines + zkx_local_defines(),
        linkopts = linkopts + zkx_linkopts(),
        alwayslink = alwayslink,
        **kwargs
    )

def zkx_cc_binary(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        linkopts = [],
        safe_code = True,
        force_exceptions = False,
        force_rtti = False,
        **kwargs):
    cc_binary(
        name = name,
        copts = copts + zkx_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions, force_rtti = force_rtti),
        defines = defines + zkx_defines(),
        local_defines = local_defines + zkx_local_defines(),
        linkopts = linkopts + zkx_linkopts(),
        **kwargs
    )

def zkx_cc_test(
        name,
        copts = [],
        defines = [],
        local_defines = [],
        linkopts = [],
        linkstatic = True,
        deps = [],
        safe_code = True,
        force_exceptions = False,
        force_rtti = False,
        **kwargs):
    cc_test(
        name = name,
        copts = copts + zkx_cxxopts(safe_code = safe_code, force_exceptions = force_exceptions, force_rtti = force_rtti),
        defines = defines + zkx_defines(),
        local_defines = local_defines + zkx_local_defines(),
        linkopts = linkopts + zkx_linkopts(),
        linkstatic = linkstatic,
        deps = deps + ["@com_google_googletest//:gtest_main"],
        **kwargs
    )

def zkx_cc_unittest(
        name,
        size = "small",
        **kwargs):
    zkx_cc_test(
        name = name,
        size = size,
        **kwargs
    )
