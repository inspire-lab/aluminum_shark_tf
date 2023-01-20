#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_ARG_UTILS_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_ARG_UTILS_H
#ifndef ALUMINUM_SHARK_COMMON_ARG_UTILS_H
#define ALUMINUM_SHARK_COMMON_ARG_UTILS_H

#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/he_backend/he_backend.h"

namespace aluminum_shark {

std::string args_to_string(const std::vector<aluminum_shark_Argument>& args);

std::string arg_to_string(const aluminum_shark_Argument& arg);

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_COMMON_ARG_UTILS_H */

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_ARG_UTILS_H \
        */
