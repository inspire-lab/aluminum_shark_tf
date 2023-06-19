#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_ENV_VARS_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_ENV_VARS_H

#include <cstdint>
#include <string>

namespace aluminum_shark {

// memory clean up setting
extern const int64_t agressive_memory_cleanup;

// symbolic computation
extern const bool symbolic_computation;

// symbolic computation file prefix
extern const std::string symbolic_computation_file_name;

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_ENV_VARS_H \
        */
