#include "tensorflow/compiler/plugin/aluminum_shark/utils/env_vars.h"

#include <cstdlib>
#include <string>

namespace aluminum_shark {

const int64_t agressive_memory_cleanup =
    std::getenv("ALUMINUM_SHARK_AGRESSIVE_MEMORY_CLEANUP") == nullptr
        ? -1
        : std::stoi(std::getenv("ALUMINUM_SHARK_AGRESSIVE_MEMORY_CLEANUP"));

const bool symbolic_computation =
    std::getenv("ALUMINUM_SHARK_SYMBOLIC_COMPUTATION") == nullptr
        ? false
        : std::stoi(std::getenv("ALUMINUM_SHARK_SYMBOLIC_COMPUTATION")) == 1;

const std::string symbolic_computation_file_name =
    std::getenv("ALUMINUM_SHARK_SYMBOLIC_COMPUTATION") == nullptr
        ? ""
        : std::getenv("ALUMINUM_SHARK_SYMBOLIC_COMPUTATION");

}  // namespace aluminum_shark
