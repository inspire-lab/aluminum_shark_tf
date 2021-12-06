// Registers the XLA_CPU device, which is an XlaDevice instantiation that runs
// operators using XLA via the XLA "Host" (CPU) backend.

#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/plugin/aluminum_shark/aluminum_shark_def.h"
// #include "tensorflow/compiler/plugin/aluminum_shark/xla_ops.h"
// #include "tensorflow/compiler/jit/xla_compile_on_demand_op.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"

namespace tf = tensorflow;

namespace aluminum_shark {

class AluminumSharkuDeviceFactory : public tf::DeviceFactory {
 public:
  tf::Status ListPhysicalDevices(std::vector<std::string>* devices) override;
  tf::Status CreateDevices(
      const tf::SessionOptions& options, const std::string& name_prefix,
      std::vector<std::unique_ptr<tf::Device>>* devices) override;
};

tf::Status AluminumSharkuDeviceFactory::ListPhysicalDevices(
    std::vector<std::string>* devices) {
  tf::XlaDeviceFlags* flags = tf::GetXlaDeviceFlags();
  if (!flags->tf_xla_enable_xla_devices) {
    VLOG(1) << "Not creating XLA devices, tf_xla_enable_xla_devices not set";
    return tf::Status::OK();
  }

  devices->push_back(absl::StrCat("/physical_device:", DEVICE_XLA_HE, ":0"));
  return tf::Status::OK();
}

tf::Status AluminumSharkuDeviceFactory::CreateDevices(
    const tf::SessionOptions& session_options, const std::string& name_prefix,
    std::vector<std::unique_ptr<tf::Device>>* devices) {
  tf::XlaDeviceFlags* flags = tf::GetXlaDeviceFlags();
  if (!flags->tf_xla_enable_xla_devices) {
    VLOG(1) << "Not creating XLA devices, tf_xla_enable_xla_devices not set";
    return tf::Status::OK();
  }
  bool compile_on_demand = flags->tf_xla_compile_on_demand;

  tf::XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_XLA_HE_JIT;
  registration.autoclustering_policy =
      compile_on_demand
          ? tf::XlaOpRegistry::AutoclusteringPolicy::kIfExplicitlyRequested
          : tf::XlaOpRegistry::AutoclusteringPolicy::kAlways;
  registration.cluster_resource_variable_ops_unsafely = true;
  registration.cluster_stack_ops = false;
  registration.cluster_tensor_array_ops = true;
  registration.cluster_stateful_rng_ops = true;
  registration.cluster_control_trigger = true;
  registration.elide_assert_and_checknumerics = true;
  registration.cluster_variant_ops = true;
  registration.cluster_slow_ops = true;
  registration.cluster_inaccurate_ops = true;
  tf::XlaOpRegistry::RegisterCompilationDevice(aluminum_shark::DEVICE_XLA_HE,
                                               registration);

  // static tf::XlaDeviceOpRegistrations* registrations =
  //     tf::RegisterXlaDeviceKernels(aluminum_shark::DEVICE_XLA_HE,
  //                                  aluminum_shark::DEVICE_XLA_HE_JIT);
  // (void)registrations;

  TF_ASSIGN_OR_RETURN(
      auto platform,
      stream_executor::MultiPlatformManager::PlatformWithName("AluminumShark"));

  tf::XlaDevice::Options options;
  options.platform = platform;
  options.device_name_prefix = name_prefix;
  options.device_name = DEVICE_XLA_HE;
  options.device_ordinal = 0;
  options.compilation_device_name = DEVICE_XLA_HE_JIT;
  options.use_multiple_streams = false;
  auto device = absl::make_unique<tf::XlaDevice>(session_options, options);

  // Setting GpuDeviceInfo because eager runtime relies on the device
  // context in tensorflow_gpu_device_info(). Also,
  // tensorflow_gpu_device_info() == nullptr is used as an IsCPU test.
  // We need XlaCpuDevice to be treated not as CPU because it allocates
  // XlaTensors, not regular Tensors.
  tf::Status status = device->UseGpuDeviceInfo();
  if (!status.ok()) {
    tf::errors::AppendToMessage(&status, "while setting up ",
                                tf::DEVICE_GPU_XLA_JIT);
    return status;
  }
  devices->push_back(std::move(device));
  return tf::Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_HE, AluminumSharkuDeviceFactory);

}  // namespace aluminum_shark

// Kernel registrations

namespace tensorflow {

REGISTER_XLA_LAUNCH_KERNEL(aluminum_shark::DEVICE_XLA_HE, XlaLocalLaunchOp,
                           aluminum_shark::kAllXlaHETypes);
REGISTER_XLA_COMPILE_KERNEL(aluminum_shark::DEVICE_XLA_HE, XlaCompileOp,
                            aluminum_shark::kAllXlaHETypes);
REGISTER_XLA_RUN_KERNEL(aluminum_shark::DEVICE_XLA_HE, XlaRunOp,
                        aluminum_shark::kAllXlaHETypes);

REGISTER_XLA_DEVICE_KERNELS(aluminum_shark::DEVICE_XLA_HE,
                            aluminum_shark::kAllXlaHETypes);

}  // namespace tensorflow
