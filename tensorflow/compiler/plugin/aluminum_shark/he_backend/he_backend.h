#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_HE_BACKEND_HE_BACKEND_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_HE_BACKEND_HE_BACKEND_H

#include <memory>
#include <string>
#include <vector>

#define ALUMINUM_SHARK_API_VERSION "0.3.0"
#define ALUMINUM_SHARK_API_VERSION_MAJOR 0
#define ALUMINUM_SHARK_API_VERSION_MINOR 3
#define ALUMINUM_SHARK_API_VERSION_PATCH 0

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

  virtual const std::string& name() = 0;
  virtual const std::string& to_string() = 0;
  virtual const API_VERSION& api_version() = 0;

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
  virtual HECtxt* encrypt(std::vector<long>& plain,
                          const std::string name = "") const = 0;
  virtual HECtxt* encrypt(std::vector<double>& plain,
                          const std::string name = "") const = 0;
  virtual HECtxt* encrypt(HEPtxt* ptxt, const std::string name = "") const = 0;

  // decryption functions
  virtual std::vector<long> decryptLong(HECtxt* ctxt) const = 0;
  virtual std::vector<double> decryptDouble(HECtxt* ctxt) const = 0;
  // for convience decryption functions are also availabe fore plaintexts. but
  // they just forward to decode
  virtual std::vector<long> decryptLong(HEPtxt* ptxt) const = 0;
  virtual std::vector<double> decryptDouble(HEPtxt* ptxt) const = 0;

  // Plaintext related

  // encoding
  virtual HEPtxt* encode(const std::vector<long>& plain) const = 0;
  virtual HEPtxt* encode(const std::vector<double>& plain) const = 0;

  // creates plaintext objects that need to be encoded on demand
  virtual HEPtxt* createPtxt(const std::vector<long>& vec) const = 0;
  virtual HEPtxt* createPtxt(const std::vector<double>& vec) const = 0;

  // decoding
  virtual std::vector<long> decodeLong(HEPtxt*) const = 0;
  virtual std::vector<double> decodeDouble(HEPtxt*) const = 0;

  virtual HE_SCHEME scheme() const = 0;

 private:
  friend HEBackend;
};

// HE plaintext
class HEPtxt {
 public:
  virtual ~HEPtxt(){};

  virtual std::string& to_string() const = 0;

  virtual const HEContext* getContext() const = 0;

  // Ptxt and Ptxt
  virtual HEPtxt* operator+(const HEPtxt* other) = 0;
  virtual HEPtxt* addInPlace(const HEPtxt* other) = 0;

  virtual HEPtxt* operator-(const HEPtxt* other) = 0;
  virtual HEPtxt* subInPlace(const HEPtxt* other) = 0;

  virtual HEPtxt* operator*(const HEPtxt* other) = 0;
  virtual HEPtxt* multInPlace(const HEPtxt* other) = 0;

  //  plain and ctxt
  // no inplace operations since they need to return a ctxt
  virtual HECtxt* operator+(const HECtxt* other) = 0;
  virtual HECtxt* operator-(const HECtxt* other) = 0;
  virtual HECtxt* operator*(const HECtxt* other) = 0;

  // integral types
  // addition
  virtual HEPtxt* operator+(long other) = 0;
  virtual HEPtxt* addInPlace(long other) = 0;
  virtual HEPtxt* operator+(double other) = 0;
  virtual HEPtxt* addInPlace(double other) = 0;

  // subtraction
  virtual HEPtxt* operator-(long other) = 0;
  virtual HEPtxt* subInPlace(long other) = 0;
  virtual HEPtxt* operator-(double other) = 0;
  virtual HEPtxt* subInPlace(double other) = 0;

  // multiplication
  virtual HEPtxt* operator*(long other) = 0;
  virtual HEPtxt* multInPlace(long other) = 0;
  virtual HEPtxt* operator*(double other) = 0;
  virtual HEPtxt* multInPlace(double other) = 0;

  virtual HEPtxt* deepCopy() = 0;

 private:
  friend HEContext;
};

// HE Ciphertext
class HECtxt {
 public:
  virtual ~HECtxt(){};

  virtual std::string to_string() const = 0;

  virtual const HEContext* getContext() const = 0;

  virtual HECtxt* deepCopy() = 0;

  // arithmetic operations

  // ctxt and ctxt
  virtual HECtxt* operator+(const HECtxt* other) = 0;
  virtual HECtxt* addInPlace(const HECtxt* other) = 0;

  virtual HECtxt* operator-(const HECtxt* other) = 0;
  virtual HECtxt* subInPlace(const HECtxt* other) = 0;

  virtual HECtxt* operator*(const HECtxt* other) = 0;
  virtual HECtxt* multInPlace(const HECtxt* other) = 0;

  // ctxt and plain

  // addition
  virtual HECtxt* operator+(HEPtxt* other) = 0;
  virtual HECtxt* addInPlace(HEPtxt* other) = 0;
  virtual HECtxt* operator+(long other) = 0;
  virtual HECtxt* addInPlace(long other) = 0;
  virtual HECtxt* operator+(double other) = 0;
  virtual HECtxt* addInPlace(double other) = 0;

  // subtraction
  virtual HECtxt* operator-(const HEPtxt* other) = 0;
  virtual HECtxt* subInPlace(const HEPtxt* other) = 0;
  virtual HECtxt* operator-(long other) = 0;
  virtual HECtxt* subInPlace(long other) = 0;
  virtual HECtxt* operator-(double other) = 0;
  virtual HECtxt* subInPlace(double other) = 0;

  // multiplication
  virtual HECtxt* operator*(HEPtxt* other) = 0;
  virtual HECtxt* multInPlace(HEPtxt* other) = 0;
  virtual HECtxt* operator*(long other) = 0;
  virtual HECtxt* multInPlace(long other) = 0;
  virtual HECtxt* operator*(double other) = 0;
  virtual HECtxt* multInPlace(double other) = 0;

  // Rotate
  virtual HECtxt* rotate(int steps) = 0;
  virtual HECtxt* rotInPlace(int steps) = 0;

 private:
  friend HEContext;
};

std::shared_ptr<HEBackend> loadBackend(const std::string& lib_path);

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_HE_BACKEND_HE_BACKEND_H \
        */
