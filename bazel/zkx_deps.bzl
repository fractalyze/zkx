load("@zkx//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("@zkx//third_party/eigen3:workspace.bzl", eigen3 = "repo")
load("@zkx//third_party/farmhash:workspace.bzl", farmhash = "repo")
load("@zkx//third_party/gloo:workspace.bzl", gloo = "repo")
load("@zkx//third_party/llvm:workspace.bzl", llvm = "repo")
load("@zkx//third_party/omp:omp_configure.bzl", "omp_configure")
load("@zkx//third_party/uv:workspace.bzl", uv = "repo")

def zkx_deps():
    omp_configure(name = "local_config_omp")

    eigen3()
    farmhash()
    gloo()
    uv()

    # Load the raw llvm-project.  llvm does not have build rules set up by default,
    # but provides a script for setting up build rules via overlays.
    llvm("llvm-raw")

    # TODO(chokobole): Delete after removing `build --noenable_bzlmod` from .bazelrc
    tf_http_archive(
        name = "bazel_skylib",
        sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        ],
    )

    # Needed by com_google_googletest
    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
        strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
        system_build_file = "@zkx//third_party/systemlibs:re2.BUILD",
        urls = tf_mirror_urls("https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_googletest",
        sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
        strip_prefix = "googletest-release-1.12.1",
        urls = tf_mirror_urls("https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "493d9905aa09124c2f44268b66205dd013f3925a7e82995f36745974e97af609",
        strip_prefix = "grpc-1.63.0",
        patch_file = ["@zkx//third_party/grpc:grpc.patch"],
        urls = tf_mirror_urls("https://github.com/grpc/grpc/archive/v1.63.0.tar.gz"),
    )

    # Needed by com_google_protobuf
    tf_http_archive(
        name = "zlib",
        build_file = "//third_party:zlib.BUILD",
        sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
        strip_prefix = "zlib-1.3.1",
        system_build_file = "@zkx//third_party/systemlibs:zlib.BUILD",
        urls = tf_mirror_urls("https://zlib.net/zlib-1.3.1.tar.gz"),
    )

    tf_http_archive(
        name = "com_github_tencent_rapidjson",
        build_file = "//third_party:rapidjson/rapidjson.BUILD",
        sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
        strip_prefix = "rapidjson-1.1.0",
        urls = tf_mirror_urls("https://github.com/Tencent/rapidjson/archive/v1.1.0.zip"),
    )

    ZKIR_COMMIT = "ea8ad988ea4bf51f8a8557c0f7fa577b795f46fa"
    tf_http_archive(
        name = "zkir",
        sha256 = "e0e4b02dbbaad9ade878b3748ae9bc0131a22a43b7b574243da2c3224b07bda6",
        strip_prefix = "zkir-{commit}".format(commit = ZKIR_COMMIT),
        urls = tf_mirror_urls("https://github.com/zk-rabbit/zkir/archive/{commit}.tar.gz".format(commit = ZKIR_COMMIT)),
    )
    # Uncomment this for development!
    # native.local_repository(
    #     name = "zkir",
    #     path = "../zkir",
    # )
