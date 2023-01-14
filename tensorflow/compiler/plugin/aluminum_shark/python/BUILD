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
    name = "arg_utils",
    srcs = ["arg_utils.cc"],
    hdrs = ["arg_utils.h"],
    deps = ["//tensorflow/compiler/plugin/aluminum_shark/he_backend"],
)

cc_library(
    name = "python_handle",
    srcs = ["python_handle.cc"],
    hdrs = ["python_handle.h"],
    deps = [
        ":arg_utils",
        "//tensorflow/compiler/plugin/aluminum_shark:ctxt",
        "//tensorflow/compiler/plugin/aluminum_shark:logging",
        "//tensorflow/compiler/plugin/aluminum_shark/he_backend",
        "//tensorflow/compiler/plugin/aluminum_shark/utils",
    ],
)
