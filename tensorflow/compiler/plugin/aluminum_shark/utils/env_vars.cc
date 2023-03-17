#include "tensorflow/compiler/plugin/aluminum_shark/utils/env_vars.h"

#include <cstdlib>
#include <string>

namespace aluminum_shark {

const int64_t agressive_memory_cleanup =
    std::getenv("ALUMINUM_SHARK_AGRESSIVE_MEMORY_CLEANUP") == nullptr
        ? -1
        : std::stoi(std::getenv("ALUMINUM_SHARK_AGRESSIVE_MEMORY_CLEANUP"));

}  // namespace aluminum_shark
