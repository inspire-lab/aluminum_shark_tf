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
    name = "aluminum_shark_transfer_manager",
    srcs = ["aluminum_shark_transfer_manager.cc"],
    hdrs = ["aluminum_shark_transfer_manager.h"],
    deps = [
        "//tensorflow/compiler/plugin/aluminum_shark:platform_id",
        "//tensorflow/compiler/xla/service:generic_transfer_manager",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = True,  # Contains per-platform transfer manager registration
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    alwayslink = 1,
)

cc_library(
    name = "layout",
    srcs = ["layout.cc"],
    hdrs = [
        "base_txt.h",
        "ctxt.h",
        "layout.h",
        "ptxt.h",
    ],
    deps = [
        ":logging",
        "//tensorflow/compiler/plugin/aluminum_shark/he_backend",
        "//tensorflow/compiler/plugin/aluminum_shark/utils",
    ],
)

cc_library(
    name = "base_txt",
    srcs = ["base_txt.cc"],
    hdrs = ["base_txt.h"],
    deps = [
        ":layout",
        ":logging",
    ],
)

cc_library(
    name = "ptxt",
    srcs = ["ptxt.cc"],
    hdrs = [
        "ctxt.h",
        "ptxt.h",
    ],
    deps = [
        ":base_txt",
        ":logging",
        "//tensorflow/compiler/plugin/aluminum_shark/he_backend",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ctxt",
    srcs = ["ctxt.cc"],
    hdrs = [
        "ctxt.h",
    ],
    deps = [
        ":base_txt",
        ":logging",
        ":ptxt",
        "//tensorflow/compiler/plugin/aluminum_shark/he_backend",
    ],
    alwayslink = 1,
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        ":ctxt",
        ":executable",
        ":logging",
        ":platform_id",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:algebraic_simplifier",
        "//tensorflow/compiler/xla/service:cholesky_expander",
        "//tensorflow/compiler/xla/service:comparison_expander",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:dynamic_index_splitter",
        "//tensorflow/compiler/xla/service:eigh_expander",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:flatten_call_graph",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_constant_folding",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_cse",
        "//tensorflow/compiler/xla/service:hlo_dce",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_subcomputation_unification",
        "//tensorflow/compiler/xla/service:layout_assignment",
        "//tensorflow/compiler/xla/service:map_inliner",
        "//tensorflow/compiler/xla/service:qr_expander",
        "//tensorflow/compiler/xla/service:reshape_mover",
        "//tensorflow/compiler/xla/service:triangular_solve_expander",
        "//tensorflow/compiler/xla/service:while_loop_simplifier",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = True,  # Contains compiler registration
)

cc_library(
    name = "platform_id",
    srcs = ["platform_id.cc"],
    hdrs = ["platform_id.h"],
    deps = ["//tensorflow/core:stream_executor_headers_lib"] + if_static(
        ["@com_google_protobuf//:protobuf"],
        ["@com_google_protobuf//:protobuf_headers"],
    ),
)

cc_library(
    name = "executable_base",
    srcs = ["executable_base.cc"],
    hdrs = ["executable_base.h"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service:dynamic_dimension_inference",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/stream_executor:event",
        "//tensorflow/stream_executor:stream",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = ["executable.h"],
    deps = [
        ":ctxt",
        ":executable_base",
        ":executor",
        ":hlo_evaluator",
        ":logging",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_execution_profile",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:stream_executor_no_cuda",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "platform",
    srcs = ["platform.cc"],
    hdrs = ["platform.h"],
    deps = [
        ":executor",
        ":platform_id",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/core:stream_executor_headers_lib",
    ],
    alwayslink = True,  # Registers itself with the MultiPlatformManager.
)

cc_library(
    name = "executor",
    srcs = ["executor.cc"],
    hdrs = ["executor.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor_headers_lib",
        "//tensorflow/stream_executor/host:host_stream",
        "//tensorflow/stream_executor/host:host_timer",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "xla_he_backend",
    srcs = ["xla_he_backend.cc"],
    hdrs = [
        "aluminum_shark_def.h",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
    ],
)

cc_library(
    name = "xla_aluminum_shark_device",
    srcs = ["xla_aluminum_shark_device.cc"],
    hdrs = [
        "aluminum_shark_def.h",
    ],
    deps = [
        # ":xla_ops",
        "//tensorflow/compiler/jit:xla_jit_headers_lib",
        # "//tensorflow/compiler/jit:flags",
        #"//tensorflow/compiler/jit:jit_compilation_passes",
        #"//tensorflow/compiler/jit:xla_kernel_creator",  # buildcleaner: keep
        #"//tensorflow/compiler/jit/kernels:xla_ops",
        #"//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/jit:xla_device",
        "@com_google_absl//absl/memory",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        ":logging",
    ],
    alwayslink = 1,
)

# cc_library(
#     name = "xla_ops",
#     hdrs = ["xla_ops.h"],
#     srcs = ["xla_ops.cc"],
#     # visibility = ["//visibility:public"],
#     # deps = XLA_OPS_DEPS + [
#     #     ":xla_ops_no_jit_rewrite_registration",
#     #     "//tensorflow/compiler/jit:jit_compilation_passes",
#     # ],
#     deps = ["//tensorflow/core:framework"],
#     alwayslink = 1,
# )
cc_library(
    name = "hlo_evaluator",
    srcs = [
        "hlo_evaluator.cc",
        "hlo_evaluator_typed_visitor.h",
        "hlo_evaluator_typed_visitor_bfloat16.cc",
        "hlo_evaluator_typed_visitor_bool.cc",
        "hlo_evaluator_typed_visitor_complex128.cc",
        "hlo_evaluator_typed_visitor_complex64.cc",
        "hlo_evaluator_typed_visitor_double.cc",
        "hlo_evaluator_typed_visitor_float.cc",
        "hlo_evaluator_typed_visitor_half.cc",
        "hlo_evaluator_typed_visitor_int16.cc",
        "hlo_evaluator_typed_visitor_int32.cc",
        "hlo_evaluator_typed_visitor_int64.cc",
        "hlo_evaluator_typed_visitor_int8.cc",
        "hlo_evaluator_typed_visitor_uint16.cc",
        "hlo_evaluator_typed_visitor_uint32.cc",
        "hlo_evaluator_typed_visitor_uint64.cc",
        "hlo_evaluator_typed_visitor_uint8.cc",
    ],
    hdrs = ["hlo_evaluator.h"],
    deps = [
        ":ctxt",
        ":logging",
        "//tensorflow/compiler/plugin/aluminum_shark/python:python_handle",
        "//tensorflow/compiler/plugin/aluminum_shark/utils",
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:dynamic_dimension_inference",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "aluminum_shark_plugin",
    deps = [
        # ":xla_ops",
        ":ctxt",
        ":logging",
        ":compiler",
        ":aluminum_shark_transfer_manager",
        ":platform",
        ":xla_he_backend",
        ":xla_aluminum_shark_device",
        "//tensorflow/core/platform:stream_executor_no_cuda",
    ],
    alwayslink = 1,
)
