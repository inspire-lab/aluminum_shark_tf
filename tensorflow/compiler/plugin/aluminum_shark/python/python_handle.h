#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_PYTHON_HANDLE_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_PYTHON_HANDLE_H

#include <chrono>
#include <list>
#include <map>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/he_backend/he_backend.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

/*
 * Singleton Class that keeps track of c++ objects accessed via python.
 */

namespace aluminum_shark {

class PythonHandle {
 public:
  static PythonHandle& getInstance();

  /*************************/
  /* Python facing methods */
  /*************************/
  // // creates ciphtertexts from the inputs
  // // template <typename T>
  // Ctxt encrypt(const std::vector<long>& ptxts, const std::string& name);

  // // decrypts the ciphtertexts inputs
  // void decrypt(std::vector<long>* ret, const std::string& name);

  // set the ciphertext inputs for the current computation
  void setCiphertexts(std::vector<Ctxt> ctxts);

  // retrive the current computation ciphertexts results
  const Ctxt retriveCiphertextsResults();

  /*************************/
  /* C/C++ facing methods  */
  /*************************/
  const std::vector<Ctxt> getCurrentCiphertexts();

  void setCurrentResult(Ctxt& ctxt);

 private:
  PythonHandle() : ts_(std::chrono::high_resolution_clock::now()) {
    AS_LOG("Creating PythonHandle Instance");
  };

  PythonHandle(const PythonHandle&) = delete;
  PythonHandle operator=(const PythonHandle&) = delete;
  PythonHandle(PythonHandle&&) = delete;
  PythonHandle operator=(PythonHandle&&) = delete;

  // static PythonHandle instance_;
  std::vector<Ctxt> input_;
  Ctxt result_;

  std::chrono::time_point<std::chrono::high_resolution_clock> ts_;
};

}  // namespace aluminum_shark
#ifdef __cplusplus
extern "C" {
#endif

// Backend

// a light wrapper that is passed outside to python. it holds a shared_ptr ot
// the backend. this stuct is meant to be dynamically allocated and destyroyed
// via the python api belows
typedef struct aluminum_shark_HEBackend {
  std::shared_ptr<aluminum_shark::HEBackend> backend;
};

// loads the hebackend shared library that is located in `libpath`.
//
// Returns: (void*)aluminum_shark_HEBackend*
void* aluminum_shark_loadBackend(const char* libpath);

// destroys the given `aluminum_shark_HEBackend`
void aluminum_shark_destroyBackend(void* backend_ptr);

// Context

// a light wrapper that is passed outside to python. it holds a shared_ptr ot
// the context. this stuct is meant to be dynamically allocated and destyroyed
// via the C api called from python
typedef struct aluminum_shark_Context {
  std::shared_ptr<aluminum_shark::HEContext> context;
};

// creates a new CKKS context using the backend specified by `backend_ptr` wich
// is a pointer to `aluminum_shark_HEBackend`. if the backend does not support
// the scheme this function returns a `nullptr`.
//
// - poly_modulus_degree (degree of polynomial modulus)
// - coeff_modulus ([ciphertext] coefficient modulus) list of moduli
// - size_coeff_modulus number of elements in the coeff_modulus array
// - plain_modulus (plaintext modulus)
//
// Returns (void*)aluminum_shark_Context*
void* aluminum_shark_CreateContextBFV(size_t poly_modulus_degree,
                                      const int* coeff_modulus,
                                      int size_coeff_modulus,
                                      size_t plain_modulus, void* backend_ptr);

// creates a new CKKS context using the backend specified by `backend_ptr` wich
// is a pointer to `aluminum_shark_HEBackend`. if the backend does not support
// the scheme this function returns a `nullptr`.
//
// - size_coeff_modulus number of elements in the coeff_modulus array
//
// Returns (void*)aluminum_shark_Context*
void* aluminum_shark_CreateContextCKKS(size_t poly_modulus_degree,
                                       const int* coeff_modulus,
                                       int size_coeff_modulus, double scale,
                                       void* backend_ptr);

// creates a new TFHE context using the backend specified by `backend_ptr`
// wich is a pointer to `aluminum_shark_HEBackend`. if the backend does not
// support the scheme this function returns a `nullptr`.
//
// Returns (void*)aluminum_shark_Context*
// TODO: add support for this
void* aluminum_shark_CreateContextTFHE(void* backend_ptr);

// create keys. takes a `aluminum_shark_Context*` which will hold a reference to
// the key
void aluminum_shark_CreatePublicKey(void* context_ptr);
void aluminum_shark_CreatePrivateKey(void* context_ptr);

// load and save the respective keys to and from files. takes a
// `aluminum_shark_Context*` which holds a reference to the key
void aluminum_shark_SavePublicKey(const char* file, void* context_ptr);
void aluminum_shark_SavePrivateKey(const char* file, void* context_ptr);
void aluminum_shark_LoadPublicKey(const char* file, void* context_ptr);
void aluminum_shark_LoadPrivateKey(const char* file, void* context_ptr);

// get the number of slots supported by the context
size_t aluminum_shark_numberOfSlots(void* context_ptr);

// destory a context
void aluminum_shark_DestroyContext(void* context_ptr);

// Ciphertext

// a light wrapper that is passed outside to python. it holds a shared_ptr ot
// the ciphertext. this stuct is meant to be dynamically allocated and
// destyroyed via the python api belows
typedef struct aluminum_shark_Ctxt {
  std::shared_ptr<aluminum_shark::Ctxt> ctxt;
};

// create a cipher from the given input values using the context. this function
// dynamically allocates a `aluminum_shark_Context`. the returned refrence needs
// to be cleaned up using `aluminum_shark_DestroyCiphertext`
//
// Returns: (void*)aluminum_shark_Context*
void* aluminum_shark_encryptLong(const long* values, int size, const char* name,
                                 void* context_ptr);
void* aluminum_shark_encryptDouble(const double* values, int size,
                                   const char* name, void* context_ptr);

// decrypts the ctxt using the given context. the result will be written into a
// `aluminum_shark_List` struct. This function always decrypts the maximum
// number of slots supported by the scheme. `size`, if positive, gives an
// indication about who many values are meaningful
void aluminum_shark_decryptLong(long* ret, int* ret_size, void* ctxt_ptr,
                                void* context_ptr);
void aluminum_shark_decryptDouble(double* ret, int* ret_size, void* ctxt_ptr,
                                  void* context_ptr);

// destroy a ciphertext
void aluminum_shark_DestroyCiphertext(void* ctxt_ptr);

// Glue code for passing data back and forth between python and c++

// set the ciphertexts used for the next computation. takes an array of pointers
// to `aluminum_shark_Ctxt` and the number of elements in that array
void aluminum_shark_SetChipherTexts(void* values, const int size);

// Retrieve the result of the most recent computaiton. returns a pointer to a
// `aluminum_shark_Ctxt`. The context the Ctxt belongs to gets written into
// context_return as pointer to `aluminum_shark_Context`
void* aluminum_shark_GetChipherTextResult(void** context_return);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_PYTHON_HANDLE_H \
        */
