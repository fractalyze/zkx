load("@rules_cc//cc:defs.bzl", "cc_library")
load("//bazel:zkx_cc.bzl", "zkx_cc_unittest")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "int_type",
    hdrs = ["int_type.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/hash",
    ],
)

zkx_cc_unittest(
    name = "tsl_lib_gtl_tests",
    srcs = [
        "int_type_test.cc",
    ],
    deps = [
        ":int_type",
    ],
)
