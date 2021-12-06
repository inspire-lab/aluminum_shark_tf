#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

#include <stdlib.h>

#include <iostream>

// read environment variable to see if we should be logging;
static bool log_on =
    std::getenv("ALUMINUM_SHARK_LOGGING") == nullptr
        ? false
        : std::stoi(std::getenv("ALUMINUM_SHARK_LOGGING")) == 1;

namespace aluminum_shark {

void log(const char* file, int line, const std::string message) {
  if (!log_on) {
    return;
  }
  std::cout << "Aluminum Shark: " << file << ":" << line << "] " << message
            << std::endl;
}

}  // namespace aluminum_shark

static bool init() {
  std::string is_on = log_on ? "ON" : "OFF";
  std::cout << "ALUMINUM_SHARK_LOGGING is " << is_on << std::endl;
  AS_LOG("logging works!");
  return true;
}

static bool loggin_init = init();