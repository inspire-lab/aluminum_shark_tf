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
    name = "python_handle",
    srcs = ["python_handle.cc"],
    hdrs = ["python_handle.h"],
    deps = [
        "//tensorflow/compiler/plugin/aluminum_shark:ctxt",
        "//tensorflow/compiler/plugin/aluminum_shark:logging",
        "//tensorflow/compiler/plugin/aluminum_shark/he_backend",
    ],
)
