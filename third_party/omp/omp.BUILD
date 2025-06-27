load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

HEADERS = [
    "omp.h",
    "ompx.h",
    "omp-tools.h",
    "ompt.h",
]

# NOTE: This is only for macos.
#       On other platforms openmp should work without local config.
genrule(
    name = "link_headers",
    outs = HEADERS,
    cmd = select({
        "@zkx//:macos_x86_64": """
      mkdir -p $(@D)/
      for file in $(OUTS); do
        file=$${file##*/}
        ln -sf /usr/local/include/$$file $(@D)/$$file
      done
    """,
        "@zkx//:macos_aarch64": """
      mkdir -p $(@D)/
      for file in $(OUTS); do
        file=$${file##*/}
        ln -sf /opt/homebrew/opt/libomp/include/$$file $(@D)/$$file
      done
    """,
        "//conditions:default": "",
    }),
)

cc_library(
    name = "omp",
    hdrs = HEADERS,
    include_prefix = "third_party/omp/include",
    includes = ["."],
    linkopts = select({
        "@zkx//:macos_x86_64": ["-L/usr/local/lib"],
        "@zkx//:macos_aarch64": ["-L/opt/homebrew/opt/libomp/lib"],
        "//conditions:default": [],
    }) + [
        "-lomp",
    ],
)
