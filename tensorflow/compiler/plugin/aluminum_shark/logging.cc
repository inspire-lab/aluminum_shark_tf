#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

#include <stdlib.h>

#include <iostream>

namespace {
// read environment variable to see if we should be logging;
bool log_on = std::getenv("ALUMINUM_SHARK_LOGGING") == nullptr
                  ? false
                  : std::stoi(std::getenv("ALUMINUM_SHARK_LOGGING")) == 1;
}  // namespace

namespace aluminum_shark {

void log(const char* file, int line, const std::string message) {
  if (!log_on) {
    return;
  }
  std::cout << "Aluminum Shark: " << file << ":" << line << "] " << message
            << std::endl;
}

void enable_logging(bool activate) {
  if (!activate) {
    AS_LOG("Logging turned off");
  }
  log_on = activate;
  AS_LOG("Logging turned on");  // only is logged if it actually was turned on
}

bool log() { return log_on; }

NullStream& nullstream() {
  static NullStream nullstream;
  return nullstream;
}

}  // namespace aluminum_shark

static bool init() {
  std::string is_on = log_on ? "ON" : "OFF";
  std::cout << "ALUMINUM_SHARK_LOGGING is " << is_on << std::endl;
  AS_LOG("logging works!");
  return true;
}

static bool loggin_init = init();