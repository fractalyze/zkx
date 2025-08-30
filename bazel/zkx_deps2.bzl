"""ZKX dependencies 2."""

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    # TODO(chokobole): Uncomment this when we need cuDNN.
    # "CUDNN_REDISTRIBUTIONS",
)
load("@version//:lastchange.bzl", "lastchange_setup")
load("@zkx//third_party/gpus/cuda/hermetic:cuda_configure.bzl", "cuda_configure")
load(
    "@zkx//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    # TODO(chokobole): Uncomment this when we need cuDNN.
    # "cudnn_redist_init_repository",
)
load("@zkx//third_party/llvm:setup.bzl", "llvm_setup")

def zkx_deps2():
    # Load the raw llvm-project.  llvm does not have build rules set up by default,
    # but provides a script for setting up build rules via overlays.
    llvm_setup(name = "llvm-project")
    lastchange_setup(name = "lastchange")

    grpc_deps()

    cuda_redist_init_repositories(
        cuda_redistributions = CUDA_REDISTRIBUTIONS,
    )

    # TODO(chokobole): Uncomment this when we need cuDNN.
    # cudnn_redist_init_repository(
    #     cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
    # )

    cuda_configure(name = "local_config_cuda")
