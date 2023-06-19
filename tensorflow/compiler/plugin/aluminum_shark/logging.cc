#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

#include <stdlib.h>

#include <iostream>

bool AS_LOG_TF = false;

namespace {
// read environment variable to see if we should be logging;
bool log_on = std::getenv("ALUMINUM_SHARK_LOGGING") == nullptr
                  ? false
                  : std::stoi(std::getenv("ALUMINUM_SHARK_LOGGING")) == 1;
int log_level = aluminum_shark::AS_WARNING;

std::string log_prefix = "Aluminum Shark";
}  // namespace

namespace aluminum_shark {

// logging singelton

// Log& Log::getInstance() {
//   static Log instance_;
//   return instance_;
// }

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

bool log(int level) { return level >= log_level; }

// TODO:
bool log_large_vectors() { return false; };

void set_log_level(int level) { log_level = level; }
int get_log_level() { return log_level; }

void set_log_prefix(const std::string& prefix) { log_prefix = prefix; }

const std::string& get_log_prefix() { return log_prefix; }

NullStream& nullstream() {
  static NullStream nullstream;
  return nullstream;
}

}  // namespace aluminum_shark

static bool init() {
  std::string is_on = log_on ? "ON" : "OFF";
  // std::cout << "ALUMINUM_SHARK_LOGGING is " << is_on << std::endl;
  AS_LOG("logging works!");
  return true;
}

static bool loggin_init = init();