load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "he_backend",
    srcs = ["he_backend.cc"],
    hdrs = ["he_backend.h"],
    deps = [
        "//tensorflow/compiler/plugin/aluminum_shark:logging"
        ],
)

