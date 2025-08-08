""" GPU-specific build macros.
"""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda_is_configured")

def gpu_kernel_library(name, copts = [], local_defines = [], tags = [], **kwargs):
    cuda_library(
        name = name + "_cuda",
        local_defines = local_defines + if_cuda_is_configured(["GOOGLE_CUDA=1"]),
        copts = copts,
        tags = ["manual"] + tags,
        **kwargs
    )

    # TODO(chokobole): Uncomment this. Dependency: rocm_library
    # rocm_library(
    #     name = name + "_rocm",
    #     local_defines = local_defines + if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    #     copts = copts + rocm_copts(),
    #     tags = ["manual"] + tags,
    #     **kwargs
    # )
    native.alias(
        name = name,
        # TODO(chokobole): Uncomment this. Dependency: if_rocm_is_configured
        # actual = if_rocm_is_configured(":%s_rocm" % name, "%s_cuda" % name),
        actual = "%s_cuda" % name,
        tags = ["gpu"] + tags,
    )
