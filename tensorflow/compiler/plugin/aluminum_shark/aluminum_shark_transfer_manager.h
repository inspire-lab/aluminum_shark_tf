#ifndef TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_ALUMINUM_SHARK_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_ALUMINUM_SHARK_TRANSFER_MANAGER_H_

#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// An implementation of the XLA GenericTransferManager for the HE backend.
class AluminumSharkTransferManager : public GenericTransferManager {
 public:
  AluminumSharkTransferManager();
  ~AluminumSharkTransferManager() override = default;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AluminumSharkTransferManager);
};

}  // namespace xla

#endif /* TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_ALUMINUM_SHARK_TRANSFER_MANAGER_H_ */
