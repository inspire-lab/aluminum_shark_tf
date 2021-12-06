#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LOGGING_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LOGGING_H

#include <string>

namespace aluminum_shark {

void log(const char* file, int line, const std::string message);

}  // namespace aluminum_shark

#define AS_LOG(msg) ::aluminum_shark::log(__FILE__, __LINE__, msg)

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LOGGING_H \
        */
