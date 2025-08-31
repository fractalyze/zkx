"""ZKX dependencies 3."""

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

def zkx_deps3():
    """ZKX dependencies."""

    grpc_extra_deps()
    protobuf_deps()
