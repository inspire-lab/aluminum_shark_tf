#include "tensorflow/compiler/plugin/aluminum_shark/aluminum_shark_transfer_manager.h"

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/aluminum_shark/platform_id.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"

namespace xla {

AluminumSharkTransferManager::AluminumSharkTransferManager()
    : GenericTransferManager(se::aluminum_shark::kXlaAluminumSharkPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

}  // namespace xla

static std::unique_ptr<xla::TransferManager>
CreateAluminumSharkTransferManager() {
  return absl::make_unique<xla::AluminumSharkTransferManager>();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(
      stream_executor::aluminum_shark::kXlaAluminumSharkPlatformId,
      &CreateAluminumSharkTransferManager);
  return true;
}

static bool module_initialized = InitModule();
