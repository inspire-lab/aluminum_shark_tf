#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_HE_BACKEND_HE_BACKEND_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_HE_BACKEND_HE_BACKEND_H

#include <memory>
#include <string>
#include <vector>

#define ALUMINUM_SHARK_API_VERSION "1.0.0"
#define ALUMINUM_SHARK_API_VERSION_MAJOR 1
#define ALUMINUM_SHARK_API_VERSION_MINOR 0
#define ALUMINUM_SHARK_API_VERSION_PATCH 0

extern "C" {
// struct to transport data between python and c++.
struct aluminum_shark_Argument {
  const char* name;
  // 0: int
  // 1: double
  // 2: string
  uint type;

  // if true the `array_` member will point to an array containing `size_`
  // elements of `type`.
  bool is_array = false;

  // data holding variables
  long int_;
  double double_;
  const char* string_;
  // holds data if `is_array` == ture.
  void* array_ = nullptr;
  size_t size_;
};
}  // extern "C"

namespace aluminum_shark {

struct API_VERSION {
  const size_t major = ALUMINUM_SHARK_API_VERSION_MAJOR;
  const size_t minor = ALUMINUM_SHARK_API_VERSION_MINOR;
  const size_t patch = ALUMINUM_SHARK_API_VERSION_PATCH;
};

enum HE_SCHEME { BFV = 0, CKKS, TFHE };

class HEContext;
class HECtxt;
class HEPtxt;
class Monitor;

// This class wraps around an externally implemented HE backend. The backend is
// implmented in its own shared library which is loaded dynamically. The backend
// needs to have a function that with the following signature
// `std::shared_ptr<aluminum_shark::HEBackend> createBackend()`
class HEBackend {
 public:
  // Create an HEContect
  virtual ~HEBackend(){};
  virtual HEContext* createContextBFV(size_t poly_modulus_degree,
                                      const std::vector<int>& coeff_modulus,
                                      size_t plain_modulus) = 0;
  virtual HEContext* createContextCKKS(size_t poly_modulus_degree,
                                       const std::vector<int>& coeff_modulus,
                                       double scale) = 0;

  virtual HEContext* createContextCKKS(
      std::vector<aluminum_shark_Argument> arguments) = 0;

  virtual const std::string& name() = 0;
  virtual const std::string& to_string() = 0;
  virtual const API_VERSION& api_version() = 0;

  virtual void set_log_level(int level) = 0;

  // enables or disables the moniotr. can return nullptr. enabling an already
  // enabled moitor reutrns the same pointer and does not create a new one
  virtual std::shared_ptr<Monitor> enable_ressource_monitor(bool) const = 0;
  virtual std::shared_ptr<Monitor> get_ressource_monitor() const = 0;

 private:
  std::shared_ptr<void> lib_handle_;
  friend std::shared_ptr<HEBackend> loadBackend(const std::string& lib_path);
};

// Provides all the nessecary functions to create keys, ciphertexts, etc.
class HEContext {
 public:
  virtual ~HEContext(){};

  virtual const std::string& to_string() const = 0;

  virtual const HEBackend* getBackend() const = 0;

  virtual int numberOfSlots() const = 0;

  // Key management

  // create a public and private key
  virtual void createPublicKey() = 0;
  virtual void createPrivateKey() = 0;

  // save public key to file
  virtual void savePublicKey(const std::string& file) = 0;
  // save private key ot file
  virtual void savePrivateKey(const std::string& file) = 0;

  // load public key from file
  virtual void loadPublicKey(const std::string& file) = 0;
  // load private key from file
  virtual void loadPrivateKey(const std::string& file) = 0;

  // Ciphertext related

  // encryption Functions
  virtual std::shared_ptr<HECtxt> encrypt(
      std::vector<long>& plain, const std::string name = "") const = 0;
  virtual std::shared_ptr<HECtxt> encrypt(
      std::vector<double>& plain, const std::string name = "") const = 0;
  virtual std::shared_ptr<HECtxt> encrypt(
      std::shared_ptr<HEPtxt> ptxt, const std::string name = "") const = 0;

  // decryption functions
  virtual std::vector<long> decryptLong(std::shared_ptr<HECtxt> ctxt) const = 0;
  virtual std::vector<double> decryptDouble(
      std::shared_ptr<HECtxt> ctxt) const = 0;
  // for convience decryption functions are also availabe fore plaintexts. but
  // they just forward to decode
  virtual std::vector<long> decryptLong(std::shared_ptr<HEPtxt> ptxt) const = 0;
  virtual std::vector<double> decryptDouble(
      std::shared_ptr<HEPtxt> ptxt) const = 0;

  // Plaintext related

  // encoding
  virtual std::shared_ptr<HEPtxt> encode(
      const std::vector<long>& plain) const = 0;
  virtual std::shared_ptr<HEPtxt> encode(
      const std::vector<double>& plain) const = 0;

  // creates plaintext objects that need to be encoded on demand
  virtual std::shared_ptr<HEPtxt> createPtxt(
      const std::vector<long>& vec) const = 0;
  virtual std::shared_ptr<HEPtxt> createPtxt(
      const std::vector<double>& vec) const = 0;

  // decoding
  virtual std::vector<long> decodeLong(std::shared_ptr<HEPtxt>) const = 0;
  virtual std::vector<double> decodeDouble(std::shared_ptr<HEPtxt>) const = 0;

  virtual HE_SCHEME scheme() const = 0;

 private:
  friend HEBackend;
};

// HE plaintext
class HEPtxt {
 public:
  virtual ~HEPtxt(){};

  virtual std::string to_string() const = 0;

  virtual const HEContext* getContext() const = 0;

  virtual std::shared_ptr<HEPtxt> deepCopy() = 0;

 private:
  friend HEContext;
};

// HE Ciphertext
class HECtxt {
 public:
  virtual ~HECtxt(){};

  virtual std::string to_string() const = 0;

  virtual const HEContext* getContext() const = 0;

  virtual std::shared_ptr<HECtxt> deepCopy() = 0;

  // arithmetic operations

  // ctxt and ctxt
  virtual std::shared_ptr<HECtxt> operator+(
      const std::shared_ptr<HECtxt> other) = 0;
  virtual void addInPlace(const std::shared_ptr<HECtxt> other) = 0;

  virtual std::shared_ptr<HECtxt> operator-(
      const std::shared_ptr<HECtxt> other) = 0;
  virtual void subInPlace(const std::shared_ptr<HECtxt> other) = 0;

  virtual std::shared_ptr<HECtxt> operator*(
      const std::shared_ptr<HECtxt> other) = 0;
  virtual void multInPlace(const std::shared_ptr<HECtxt> other) = 0;

  // returns the size of the ciphertext in bytes
  virtual size_t size() = 0;

  // ctxt and plain

  // addition
  virtual std::shared_ptr<HECtxt> operator+(std::shared_ptr<HEPtxt> other) = 0;
  virtual void addInPlace(std::shared_ptr<HEPtxt> other) = 0;
  virtual std::shared_ptr<HECtxt> operator+(long other) = 0;
  virtual void addInPlace(long other) = 0;
  virtual std::shared_ptr<HECtxt> operator+(double other) = 0;
  virtual void addInPlace(double other) = 0;

  // subtraction
  virtual std::shared_ptr<HECtxt> operator-(std::shared_ptr<HEPtxt> other) = 0;
  virtual void subInPlace(std::shared_ptr<HEPtxt> other) = 0;
  virtual std::shared_ptr<HECtxt> operator-(long other) = 0;
  virtual void subInPlace(long other) = 0;
  virtual std::shared_ptr<HECtxt> operator-(double other) = 0;
  virtual void subInPlace(double other) = 0;

  // multiplication
  virtual std::shared_ptr<HECtxt> operator*(std::shared_ptr<HEPtxt> other) = 0;
  virtual void multInPlace(std::shared_ptr<HEPtxt> other) = 0;
  virtual std::shared_ptr<HECtxt> operator*(long other) = 0;
  virtual void multInPlace(long other) = 0;
  virtual std::shared_ptr<HECtxt> operator*(double other) = 0;
  virtual void multInPlace(double other) = 0;

  // Rotate
  virtual std::shared_ptr<HECtxt> rotate(int steps) = 0;
  virtual void rotInPlace(int steps) = 0;

 private:
  friend HEContext;
};

// A class to monitor the ressource consumption of the backend.
class Monitor {
 public:
  // retrieves the value specified by name and writes it into value, returns
  // false if the value is not logged or unsoproted;
  virtual bool get(const std::string& name, double& value) = 0;

  // can be used to iterate over all logged valued by this monitor. puts the
  // name of the value into `name` and the value into `value`. Returns false if
  // there are no more values. Calling it again after that restarts
  virtual bool get_next(std::string& name, double& value) = 0;

  // returns a list of all values supported by this monitor
  virtual const std::vector<std::string>& values() = 0;
};

std::shared_ptr<HEBackend> loadBackend(const std::string& lib_path);

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_HE_BACKEND_HE_BACKEND_H \
        */
