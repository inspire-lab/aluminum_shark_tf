#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_EXCEPTION_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_EXCEPTION_H
namespace aluminum_shark {

class decryption_error : public std::excpetion {
 public:
  enum REASON { OTHER, NO_KEY, WRONG_TYPE };

  decryption_error(const std::string& message, REASON reason)
      : failure_reason(reason), messgage_(message){};

  const REASON failure_reason;

  virtual const char* what() const noexcept {return message_};

 private:
  const std::string message_;
}
}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_EXCEPTION_H \
        */
