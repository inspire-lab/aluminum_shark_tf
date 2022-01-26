#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LOGGING_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LOGGING_H

#include <iostream>
#include <string>

namespace aluminum_shark {

class NullStream : public std::ostream {
 public:
  NullStream() : std::ostream(nullptr) {}
  NullStream(const NullStream&) : std::ostream(nullptr) {}
};

template <class T>
const NullStream& operator<<(NullStream&& os, const T& value) {
  return os;
}

void log(const char* file, int line, const std::string message);

void enable_logging(bool activate);

bool log();

NullStream& nullstream();

}  // namespace aluminum_shark

// log a sing string
#define AS_LOG(msg) ::aluminum_shark::log(__FILE__, __LINE__, msg)
// streaming interface
#define AS_LOG_S                                                         \
  (::aluminum_shark::log() ? std::cout : ::aluminum_shark::nullstream()) \
      << "Aluminum Shark: " << __FILE__ << ":" << __LINE__ << "] "
// append to stream
#define AS_LOG_SA \
  (::aluminum_shark::log() ? std::cout : ::aluminum_shark::nullstream())

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LOGGING_H \
        */
