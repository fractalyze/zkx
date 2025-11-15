"""Provides the repo macro to import zk_dtypes.

zk_dtypes provides ZK-specific data-types like int4.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    ZK_DTYPES_COMMIT = "795ba985e3cbda032224e679daddf64f5bb62386"
    ZK_DTYPES_SHA256 = "3d879acc3eb200be7dca3cb4577cdd49c1e4259053625158eec0fef39162c93d"
    tf_http_archive(
        name = "zk_dtypes_py",
        build_file = "//third_party/py/zk_dtypes:zk_dtypes_py.BUILD",
        link_files = {
            "//third_party/py/zk_dtypes:zk_dtypes.BUILD": "zk_dtypes/BUILD.bazel",
        },
        sha256 = ZK_DTYPES_SHA256,
        strip_prefix = "zk_dtypes-{commit}".format(commit = ZK_DTYPES_COMMIT),
        urls = tf_mirror_urls("https://github.com/fractalyze/zk_dtypes/archive/{commit}/zk_dtypes-{commit}.tar.gz".format(commit = ZK_DTYPES_COMMIT)),
    )
