#include "tensorflow/compiler/plugin/aluminum_shark/he_backend/he_backend.h"

#include <dlfcn.h>

#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

namespace aluminum_shark {

static constexpr API_VERSION version = API_VERSION();

std::shared_ptr<HEBackend> loadBackend(const std::string& lib_path) {
  AS_LOG_S << "Using API version: " << version.major << "." << version.minor
           << "." << version.patch << std::endl;
  AS_LOG("Loading backend: " + lib_path);
  void* raw_p = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (raw_p == 0) {
    // TODO fail here
    std::string error(dlerror());
    AS_LOG("Failed to open shared library " + lib_path +
           " Error was: " + error);
  }
  std::shared_ptr<void> ptr(raw_p, [](void* p) { dlclose(p); });

  // getting ready to call
  //  `std::shared_ptr<aluminum_shark::HEBackend> createBackend()`
  std::shared_ptr<HEBackend> (*createFunc)() =
      reinterpret_cast<std::shared_ptr<HEBackend> (*)()>(
          dlsym(raw_p, "createBackend"));
  if (createFunc == 0) {
    // TODO fail here
    AS_LOG("Failed to find `createBackend` in shared library " + lib_path);
  }
  std::shared_ptr<HEBackend> backend = createFunc();
  backend->lib_handle_ = ptr;
  const API_VERSION& backend_ver = backend->api_version();
  AS_LOG_S << "Sucessfully opened backend: " << backend->name()
           << " API version: " << backend_ver.major << "." << backend_ver.minor
           << "." << backend_ver.patch << " from " << lib_path << std::endl;
  if (version.major != backend_ver.major) {
    AS_LOG_S << "WARNING: incompatibale API versions! " << std::endl;
  }
  return backend;
}

}  // namespace aluminum_shark