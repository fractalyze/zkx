"""Provides the repo macro to import zk_dtypes.

zk_dtypes provides ZK-specific data-types like int4.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    ZK_DTYPES_COMMIT = "f7c36b3b5925c8355868adbf0b2ddb415bc089c8"
    ZK_DTYPES_SHA256 = "da21588d3f208a622cb2edf7b873c6bbcff9a105d196a0b1277f6ea8f7ce476d"
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
