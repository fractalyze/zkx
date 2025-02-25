load("@zkx//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("@zkx//third_party/absl:workspace.bzl", absl = "repo")
load("@zkx//third_party/eigen3:workspace.bzl", eigen3 = "repo")
load("@zkx//third_party/llvm:workspace.bzl", llvm = "repo")

def zkx_deps():
    absl()
    eigen3()

    # Load the raw llvm-project.  llvm does not have build rules set up by default,
    # but provides a script for setting up build rules via overlays.
    llvm("llvm-raw")

    # Needed by com_google_googletest
    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
        strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
        system_build_file = "@tsl//third_party/systemlibs:re2.BUILD",
        urls = tf_mirror_urls("https://github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_googletest",
        sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
        strip_prefix = "googletest-release-1.12.1",
        urls = tf_mirror_urls("https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz"),
    )
