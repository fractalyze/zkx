"""ZKX rules."""

# See https://semver.org/
VERSION_MAJOR = 1
VERSION_MINOR = 0
VERSION_PATCH = 0
VERSION_PRERELEASE = ""
VERSION = ".".join([str(VERSION_MAJOR), str(VERSION_MINOR), str(VERSION_PATCH)])

# Platform specific conditions
def if_android(a, b = []):
    return select({
        "@platforms//os:android": a,
        "//conditions:default": b,
    })

def if_aarch64(a, b = []):
    return select({
        "@platforms//cpu:aarch64": a,
        "//conditions:default": b,
    })

def if_arm(a, b = []):
    return select({
        "@platforms//cpu:arm": a,
        "//conditions:default": b,
    })

def if_linux(a, b = []):
    return select({
        "@platforms//os:linux": a,
        "//conditions:default": b,
    })

def if_macos(a, b = []):
    return select({
        "@platforms//os:macos": a,
        "//conditions:default": b,
    })

def if_posix(a, b = []):
    return select({
        "@platforms//os:windows": b,
        "//conditions:default": a,
    })

def if_windows(a, b = []):
    return select({
        "@platforms//os:windows": a,
        "//conditions:default": b,
    })

def if_x86_32(a, b = []):
    return select({
        "@platforms//cpu:x86_32": a,
        "//conditions:default": b,
    })

def if_x86_64(a, b = []):
    return select({
        "@platforms//cpu:x86_64": a,
        "//conditions:default": b,
    })

# Feature specific conditions
def if_has_exception(a, b = []):
    return select({
        "@zkx//:zkx_has_exception": a,
        "//conditions:default": b,
    })

def if_has_openmp(a, b = []):
    return select({
        "@zkx//:zkx_has_openmp": a,
        "//conditions:default": b,
    })

def if_has_openmp_on_macos(a, b = []):
    return select({
        "@zkx//:zkx_has_openmp_on_macos": a,
        "//conditions:default": b,
    })

def if_has_rtti(a, b = []):
    return select({
        "@zkx//:zkx_has_rtti": a,
        "//conditions:default": b,
    })
