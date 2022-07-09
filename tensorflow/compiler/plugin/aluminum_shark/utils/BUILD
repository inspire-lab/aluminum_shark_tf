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
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = ["//tensorflow/compiler/plugin/aluminum_shark:logging"],
)

cc_library(
    name = "parallel",
    srcs = ["parallel.cc"],
    hdrs = ["parallel.h"],
    deps = ["//tensorflow/compiler/plugin/aluminum_shark:logging"],
)

cc_library(
    name = "exception",
    hdrs = ["exception.h"],
)
