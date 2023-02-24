#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_PYTHON_HANDLE_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_PYTHON_HANDLE_H

#include <chrono>
#include <functional>
#include <list>
#include <map>
#include <queue>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/he_backend/he_backend.h"
#include "tensorflow/compiler/plugin/aluminum_shark/layout.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

/*
 * Singleton Class that keeps track of c++ objects accessed via python.
 */

namespace aluminum_shark {

class ComputationHandle {
 public:
  ComputationHandle(void* (*ctxt_callback)(int*),
                    void (*result_callback)(void*, int),
                    const char* forced_layout, bool clear_memory)
      : ctxt_callback_(ctxt_callback),
        result_callback_(result_callback),
        forced_layout_(forced_layout),
        clear_memory_(clear_memory){};

  // uses the python callback to get the handles
  std::vector<Ctxt> getCiphertTexts();

  // retrieve the result of the computation
  void transfereResults(std::vector<Ctxt>& ctxts);

  // used forced layout
  bool useForcedLayout() const;

  // clear memory during computation
  bool clearMemory() const;

  // use this layout for all ptxt and ctxt in the computation. can reutrn
  // `nullptr`
  const char* getForcedLayout() const;

 private:
  std::function<void*(int*)> ctxt_callback_;
  std::function<void(void*, int)> result_callback_;
  const char* forced_layout_;
  const bool clear_memory_;
};

class PythonHandle {
 public:
  static PythonHandle& getInstance();

  /*************************/
  /* Python facing methods */
  /*************************/

  // registers the handler for the next computation
  void registerComputation(std::shared_ptr<ComputationHandle> ch);

  /*************************/
  /* C/C++ facing methods  */
  /*************************/
  std::shared_ptr<ComputationHandle> consumeComputationHandle();

  // old
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

  std::queue<std::shared_ptr<ComputationHandle>> computationQueue_;

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

// create a new CKKS context using the dynamic argument API.
// accpets an array of aluminum_shark::Argument* containing count elements
//
// Returns (void*)aluminum_shark_Context*
void* aluminum_shark_CreateContextCKKS_dynamic(aluminum_shark_Argument* args,
                                               int count, void* backend_ptr);

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

// Layout

// a light wrapper that is passed outside to python. it holds a shared_ptr to
// the layout. this stuct is meant to be dynamically allocated and
// destyroyed via the python api belows
typedef struct aluminum_shark_Layout {
  std::shared_ptr<aluminum_shark::Layout> layout;
};

// get a list of all availabe layouts
const char* const* aluminum_shark_GetAvailabeLayouts(size_t* size);

// Ciphertext

// a light wrapper that is passed outside to python. it holds a shared_ptr ot
// the ciphertext. this stuct is meant to be dynamically allocated and
// destyroyed via the python api belows
typedef struct aluminum_shark_Ctxt {
  std::shared_ptr<aluminum_shark::Ctxt> ctxt;
};

size_t aluminum_shark_GetCtxtShapeLen(void* ctxt_ptr);

void aluminum_shark_GetCtxtShape(void* ctxt_ptr, size_t* shape_array);

// destroy a ciphertext
void aluminum_shark_DestroyCiphertext(void* ctxt_ptr);

// create a cipher from the given input values using the context. this function
// dynamically allocates a `aluminum_shark_Ctxt`. the returned refrence needs
// to be cleaned up using `aluminum_shark_DestroyCiphertext`
//
// Returns: (void*)aluminum_shark_Ctxt*
void* aluminum_shark_encryptLong(const long* values, int size, const char* name,
                                 const size_t* shape, int shape_size,
                                 const char* layout, void* context_ptr);
void* aluminum_shark_encryptDouble(const double* values, int size,
                                   const char* name, const size_t* shape,
                                   int shape_size, const char* layout,
                                   void* context_ptr);

// decrypts the ctxt using the given context. the result will be written into a
// `aluminum_shark_List` struct. This function always decrypts the maximum
// number of slots supported by the scheme. The array pointed at by `ret` needs
// to have allocated memory for at least as many elements as the tensor size of
// the ciphertext. (the tensor size is the product of all dimensions of the
// shape)
void aluminum_shark_decryptLong(long* ret, void* ctxt_ptr, void* context_ptr);
void aluminum_shark_decryptDouble(double* ret, void* ctxt_ptr,
                                  void* context_ptr);

// Glue code for passing data back and forth between python and c++

// a light wrapper that is passed outside to python. it holds a shared_ptr to a
// computation
typedef struct aluminum_shark_Computation {
  std::shared_ptr<aluminum_shark::ComputationHandle> computation;
};

// registert for the next computation
void* aluminum_shark_RegisterComputation(void* (*ctxt_callback)(int*),
                                         void (*result_callback)(void*, int),
                                         const char* forced_layout,
                                         bool clear_memory);

// turns logging on or off
void aluminum_shark_EnableLogging(bool on);

// sets the log level
void aluminum_shark_SetLogLevel(int level);

// sets the backend log level
void aluminum_shark_SetBackendLogLevel(int level, void* backend_ptr);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_PYTHON_HANDLE_H \
        */
