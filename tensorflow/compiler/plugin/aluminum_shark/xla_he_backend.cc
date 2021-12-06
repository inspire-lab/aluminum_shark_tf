#include "tensorflow/compiler/plugin/aluminum_shark/aluminum_shark_def.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace aluminum_shark {

const char* const DEVICE_XLA_HE = "XLA_HE";
const char* const DEVICE_XLA_HE_JIT = "XLA_HE_JIT";

bool HEOpFilter(tensorflow::KernelDef* kdef) {
  // eventually we need to filter out unsupporte OPs
  if (kdef->op() == "Const") {
    tensorflow::AddDtypeToKernelDefConstraint("dtype", tensorflow::DT_STRING,
                                              kdef);
  }
  if (kdef->op() == "Assert") {
    tensorflow::AddDtypeToKernelDefConstraint("T", tensorflow::DT_STRING, kdef);
  }
  return true;
}

// REGISTER_XLA_BACKEND(DEVICE_XLA_HE, kAllXlaHETypes, HEOpFilter);

REGISTER_XLA_BACKEND(DEVICE_XLA_HE_JIT, kAllXlaHETypes, HEOpFilter);

}  // namespace aluminum_shark