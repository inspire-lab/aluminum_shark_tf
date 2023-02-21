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

bool log_large_vectors();

void set_log_level(int level);
int get_log_level();

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

#define AS_LOG_MACRO(level, lvl_prefix)                                       \
  if (!::aluminum_shark::log(level)) {                                        \
  } else                                                                      \
    std::cout << ::aluminum_shark::get_log_prefix() << " " lvl_prefix << ": " \
              << __FILE__ << ":" << __LINE__ << "] "

#define AS_LOG_CRITICAL AS_LOG_MACRO(::aluminum_shark::AS_CRITICAL, "CRITICAL")
#define AS_LOG_ERROR AS_LOG_MACRO(::aluminum_shark::AS_ERROR, "ERROR")
#define AS_LOG_WARNING AS_LOG_MACRO(::aluminum_shark::AS_WARNING, "WARNING")
#define AS_LOG_INFO AS_LOG_MACRO(::aluminum_shark::AS_INFO, "INFO")
#define AS_LOG_DEBUG AS_LOG_MACRO(::aluminum_shark::AS_DEBUG, "DEBUG")

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LOGGING_H \
        */
