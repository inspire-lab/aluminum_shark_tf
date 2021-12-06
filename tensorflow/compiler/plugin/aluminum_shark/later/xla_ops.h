#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_KERNELS_XLA_OPS_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_KERNELS_XLA_OPS_H

#include <atomic>

#include "tensorflow/compiler/jit/xla_compilation_cache.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace tf = tensorflow;

namespace aluminum_shark {

// XlaLocalLaunchBase is almost the same as XlaLocalLaunchOp.
// The only difference is that it does not require arguments to follow
// the "constants, then regular args, then resources" order.
// It takes vectors of constant and resource arguments explicitly.
// It does not have corresponding OpDef because it is never present
// in the GraphDef.
// Currently, it is used by eager runtime. FunctionLibraryRuntime creates
// this kernel when asked to create a kernel for an XLA-compiled function.
//
// `has_ref_vars`: whether the input computation can have reference variables.
// TODO(cheshire): instead derive this information from the input graph.
class XlaLocalLaunchBase : public tf::OpKernel {
 public:
  XlaLocalLaunchBase(tf::OpKernelConstruction* ctx,
                     const std::vector<int>& constants,
                     const std::vector<int>& resources,
                     const tf::NameAttrList& function, bool has_ref_vars);
  XlaLocalLaunchBase(const XlaLocalLaunchBase&) = delete;
  XlaLocalLaunchBase& operator=(const XlaLocalLaunchBase&) = delete;
  ~XlaLocalLaunchBase() override = default;

  void Compute(tf::OpKernelContext* ctx) override;

 protected:
  // Indexes of compile-time constant inputs
  const std::vector<int> constants_;
  // Indexes of resource inputs
  const std::vector<int> resources_;

  const tf::NameAttrList function_;
  const tf::XlaPlatformInfo platform_info_;

  bool has_ref_vars_;
};

// XlaLocalLaunchOp is used to replace a region of the TensorFlow graph
// which will be compiled and executed using XLA.  The XlaLocalLaunchOp is
// responsible for handling interactions with the TensorFlow executor.
// Once all inputs are present, and their shapes are known, the op can
// use a 'XlaCompilationCache' to compile and execute code which is specific
// to the shapes of input Tensors.
// XlaLocalLaunchOp uses xla::LocalClient::Compile() and
// xla::LocalExecutable::Run(), and passes arguments into/out of XLA in device
// memory.
class XlaLocalLaunchOp : public XlaLocalLaunchBase {
 public:
  explicit XlaLocalLaunchOp(tf::OpKernelConstruction* ctx);
  ~XlaLocalLaunchOp() override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(XlaLocalLaunchOp);
};

class XlaCompileOp : public tf::OpKernel {
 public:
  explicit XlaCompileOp(tf::OpKernelConstruction* ctx);

  void Compute(tf::OpKernelContext* ctx) override;

 private:
  // Indexes of compile-time constant inputs
  const std::vector<int> constants_;
  // Indexes of resource inputs
  const std::vector<int> resources_;

  const tf::NameAttrList function_;

  tf::XlaPlatformInfo platform_info_;

  const bool must_compile_;

  // Whether the graph has TF reference variables.
  const bool has_ref_vars_;

  // cannot_compile_cluster_ is set to true if XLA returns an Unimplemented
  // error when compiling the cluster this _XlaCompile is supposed to compile.
  // If `cannot_compile_cluster_` is true then we avoid compiling this cluster
  // on any future calls to _XlaCompile.
  bool cannot_compile_cluster_ TF_GUARDED_BY(cannot_compile_cluster_mu_) =
      false;

  tf::mutex cannot_compile_cluster_mu_;
};

class XlaRunOp : public tf::OpKernel {
 public:
  explicit XlaRunOp(tf::OpKernelConstruction* ctx);

  void Compute(tf::OpKernelContext* ctx) override;

 private:
  const tf::XlaPlatformInfo platform_info_;
};

class XlaMergeOp : public tf::OpKernel {
 public:
  explicit XlaMergeOp(tf::OpKernelConstruction* ctx);

  void Compute(tf::OpKernelContext* ctx) override;
};

}  // namespace aluminum_shark

#endif  // TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_KERNELS_XLA_OPS_H
