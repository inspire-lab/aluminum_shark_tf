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

// TODO RP: finish this
// class Log : public std::basic_ostringstream<char> {
//  public:
//   static Log& getInstance();
// };

void log(const char* file, int line, const std::string message);

void enable_logging(bool activate);

bool log();
bool log(int level);

void set_log_level(int level);

void set_log_prefix(const std::string& prefix);
const std::string& get_log_prefix();

NullStream& nullstream();

constexpr int AS_CRITICAL = 50;
constexpr int AS_ERROR = 40;
constexpr int AS_WARNING = 30;
constexpr int AS_INFO = 20;
constexpr int AS_DEBUG = 10;

}  // namespace aluminum_shark

// log a sing string
#define AS_LOG(msg) ::aluminum_shark::log(__FILE__, __LINE__, msg)
// streaming interface
#define AS_LOG_S                                                         \
  (::aluminum_shark::log() ? std::cout : ::aluminum_shark::nullstream()) \
      << ::aluminum_shark::get_log_prefix() << ": " << __FILE__ << ":"   \
      << __LINE__ << "] "
// append to stream
#define AS_LOG_SA \
  (::aluminum_shark::log() ? std::cout : ::aluminum_shark::nullstream())

#define AS_LOG_CRITICAL                                                  \
  (::aluminum_shark::log(::aluminum_shark::AS_CRITICAL)                  \
       ? std::cout                                                       \
       : ::aluminum_shark::nullstream())                                 \
      << ::aluminum_shark::get_log_prefix() << " CRITICAL: " << __FILE__ \
      << ":" << __LINE__ << "] "
#define AS_LOG_ERROR                                                         \
  (::aluminum_shark::log(::aluminum_shark::AS_ERROR)                         \
       ? std::cout                                                           \
       : ::aluminum_shark::nullstream())                                     \
      << ::aluminum_shark::get_log_prefix() << " ERROR: " << __FILE__ << ":" \
      << __LINE__ << "] "
#define AS_LOG_WARNING                                                         \
  (::aluminum_shark::log(::aluminum_shark::AS_WARNING)                         \
       ? std::cout                                                             \
       : ::aluminum_shark::nullstream())                                       \
      << ::aluminum_shark::get_log_prefix() << " WARNING: " << __FILE__ << ":" \
      << __LINE__ << "] "
#define AS_LOG_INFO                                                         \
  (::aluminum_shark::log(::aluminum_shark::AS_INFO)                         \
       ? std::cout                                                          \
       : ::aluminum_shark::nullstream())                                    \
      << ::aluminum_shark::get_log_prefix() << " INFO: " << __FILE__ << ":" \
      << __LINE__ << "] "
#define AS_LOG_DEBUG                                                         \
  (::aluminum_shark::log(::aluminum_shark::AS_DEBUG)                         \
       ? std::cout                                                           \
       : ::aluminum_shark::nullstream())                                     \
      << ::aluminum_shark::get_log_prefix() << " DEBUG: " << __FILE__ << ":" \
      << __LINE__ << "] "

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LOGGING_H \
        */
