#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_BASE_TXT_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_BASE_TXT_H

#include <memory>
#include <vector>

namespace aluminum_shark {
// Abstract base class for `Ctxt` and `Ptxt`. Faciliates arithemtic operations
// between different types. It works under the assumption that we always
// return a ciphertext. if operations with only plaintext operatives become
// nessecary we need to redesign
class BaseTxt {
 public:
  virtual ~BaseTxt(){};

  virtual std::string to_string() const = 0;

  virtual std::shared_ptr<BaseTxt> operator+(const BaseTxt& other) const = 0;
  virtual std::shared_ptr<BaseTxt> operator*(const BaseTxt& other) const = 0;

  virtual BaseTxt& operator+=(const BaseTxt& other) = 0;
  virtual BaseTxt& operator*=(const BaseTxt& other) = 0;

  // integer
  virtual std::shared_ptr<BaseTxt> operator+(long other) = 0;
  virtual std::shared_ptr<BaseTxt> operator*(long other) = 0;
  virtual BaseTxt& operator+=(long other) = 0;
  virtual BaseTxt& operator*=(long other) = 0;
  // floating point
  virtual std::shared_ptr<BaseTxt> operator+(double other) = 0;
  virtual std::shared_ptr<BaseTxt> operator*(double other) = 0;
  virtual BaseTxt& operator+=(double other) = 0;
  virtual BaseTxt& operator*=(double other) = 0;
};

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_BASE_TXT_H \
        */
