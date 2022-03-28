#include "tensorflow/compiler/plugin/aluminum_shark/python/python_handle.h"

#include <iostream>
#include <map>
#include <memory>
// #include <mutex>

namespace {
static aluminum_shark::Ctxt I_AM_ERROR("I AM ERROR");
}  // namespace

namespace aluminum_shark {

PythonHandle& PythonHandle::getInstance() {
  static PythonHandle instance_;
  std::stringstream ss;
  ss << (void*)&instance_;
  AS_LOG("getting PythonHandle instance " +
         std::to_string(instance_.ts_.time_since_epoch().count()) +
         " addr: " + ss.str());
  return instance_;
}

/*************************/
/* Python facing methods */
/*************************/

// void PythonHandle::setCiphertexts(std::vector<Ctxt> ctxts) {
//   std::string log_msg = "setting ciphertexts ";
//   for (auto& ctxt : ctxts) {
//     log_msg += " " + ctxt.to_string();
//   }
//   AS_LOG(log_msg);
//   // input_.clear();
//   input_ = ctxts;
//   log_msg = "set ciphertexts ";
//   for (const auto& ctxt : input_) {
//     log_msg += " " + ctxt.to_string();
//   }
//   AS_LOG(log_msg);
//   for (auto& c : getCurrentCiphertexts()) {
//     AS_LOG(c.to_string());
//   }
// }

// const Ctxt PythonHandle::retriveCiphertextsResults() {
//   AS_LOG("retrieving result " + result_.to_string());
//   return result_;
// }

void PythonHandle::registerComputation(std::shared_ptr<ComputationHandle> ch) {
  computationQueue_.push(ch);
}

std::shared_ptr<ComputationHandle> PythonHandle::consumeComputationHandle() {
  std::shared_ptr<ComputationHandle> ret = computationQueue_.front();
  computationQueue_.pop();
  return ret;
}

/*************************/
/* C/C++ facing methods  */
/*************************/

const std::vector<Ctxt> PythonHandle::getCurrentCiphertexts() {
  std::string log_msg =
      "getting ciphertexts (" + std::to_string(input_.size()) + ") ";
  for (const auto& ctxt : input_) {
    log_msg += " " + ctxt.to_string();
  }
  AS_LOG(log_msg);
  return input_;
}

void PythonHandle::setCurrentResult(Ctxt& ctxt) {
  AS_LOG("seeting result " + ctxt.to_string());
  result_ = ctxt;
}

// Computation

std::vector<Ctxt> ComputationHandle::getCiphertTexts() {
  int num = -1;
  AS_LOG_S << "invoking ciphertext callback " << reinterpret_cast<void*>(&num)
           << std::endl;
  void* call_back_result = ctxt_callback_(&num);
  AS_LOG_S << "callback returned " << num << " input array ( "
           << call_back_result << ") " << std::endl;
  std::vector<Ctxt> ret;
  aluminum_shark_Ctxt** input_array =
      reinterpret_cast<aluminum_shark_Ctxt**>(call_back_result);
  for (size_t i = 0; i < num; i++) {
    aluminum_shark_Ctxt* as_ctxt =
        reinterpret_cast<aluminum_shark_Ctxt*>(input_array[i]);
    ret.push_back(as_ctxt->ctxt.operator*());
  }
  return ret;
}

// retrieve the result of the computation
void ComputationHandle::transfereResults(std::vector<Ctxt>& ctxts) {
  void** result = new void*[ctxts.size()];
  for (size_t i = 0; i < ctxts.size(); ++i) {
    aluminum_shark_Ctxt* ctxt = new aluminum_shark_Ctxt();
    ctxt->ctxt = std::make_shared<Ctxt>(ctxts[i]);
    result[i] = reinterpret_cast<void*>(ctxt);
  }
  // hand the results back to python
  result_callback_(result, ctxts.size());
  // python has taken over the owenership of the ciphertext handles. we can get
  // rid of the return array
  delete[] result;
}

}  // namespace aluminum_shark

/***********************/
/*  C API              */
/***********************/

// need as reverse mapping to return the context handle when the ctxt result is
// retrieved
static std::map<const aluminum_shark::HEContext*, aluminum_shark_Context*>
    context_map;

extern "C" {
void* aluminum_shark_loadBackend(const char* libpath) {
  AS_LOG("loading backend");
  aluminum_shark_HEBackend* ret = new aluminum_shark_HEBackend();
  std::shared_ptr<aluminum_shark::HEBackend> backend_ptr =
      aluminum_shark::loadBackend(libpath);
  ret->backend = backend_ptr;
  AS_LOG_S << "Backend ptr: " << reinterpret_cast<void*>(ret)
           << " wrapped obejct: " << ret->backend << std::endl;
  return ret;
}

void aluminum_shark_destroyBackend(void* backend_ptr) {
  AS_LOG_S << "Deleting backend: " << backend_ptr << std::endl;
  delete static_cast<aluminum_shark_HEBackend*>(backend_ptr);
}

// return (void*)aluminum_shark_Context*
void* aluminum_shark_CreateContextBFV(size_t poly_modulus_degree,
                                      const int* coeff_modulus,
                                      int size_coeff_modulus,
                                      size_t plain_modulus, void* backend_ptr) {
  std::shared_ptr<aluminum_shark::HEBackend> backend =
      static_cast<aluminum_shark_HEBackend*>(backend_ptr)->backend;
  aluminum_shark_Context* ret = new aluminum_shark_Context();
  ret->context =
      std::shared_ptr<aluminum_shark::HEContext>(backend->createContextBFV(
          poly_modulus_degree,
          std::vector<int>(coeff_modulus, coeff_modulus + size_coeff_modulus),
          plain_modulus));
  context_map[ret->context.get()] = ret;
  return ret;
}

void* aluminum_shark_CreateContextCKKS(size_t poly_modulus_degree,
                                       const int* coeff_modulus,
                                       int size_coeff_modulus, double scale,
                                       void* backend_ptr) {
  AS_LOG("creating CKKS backend");
  std::vector<int> coeff_modulus_vec(coeff_modulus,
                                     coeff_modulus + size_coeff_modulus);
  AS_LOG_S << "poly_modulus_degree " << std::to_string(poly_modulus_degree)
           << " coeff_modulus [";
  for (auto v : coeff_modulus_vec) {
    AS_LOG_SA << std::to_string(v) << ",";
  }
  AS_LOG_SA << "], scale " << std::to_string(scale)
            << " backend pointer: " << backend_ptr << std::endl;
  aluminum_shark_HEBackend* as_backend =
      static_cast<aluminum_shark_HEBackend*>(backend_ptr);
  AS_LOG_S << "cast successful " << reinterpret_cast<void*>(as_backend)
           << std::endl;
  std::shared_ptr<aluminum_shark::HEBackend> backend = as_backend->backend;
  AS_LOG_S << "backend " << backend << std::endl;
  aluminum_shark_Context* ret = new aluminum_shark_Context();
  ret->context =
      std::shared_ptr<aluminum_shark::HEContext>(backend->createContextCKKS(
          poly_modulus_degree,
          std::vector<int>(coeff_modulus, coeff_modulus + size_coeff_modulus),
          scale));
  AS_LOG_S << "Created new Context: " << reinterpret_cast<void*>(as_backend)
           << "wrappend context " << ret->context << std::endl;
  context_map[ret->context.get()] = ret;
  return ret;
}

void* aluminum_shark_CreateContextTFHE(void* backend_ptr) {
  AS_LOG("TFHE backend not implemented");
  return nullptr;
};

// Key Managment

void aluminum_shark_CreatePublicKey(void* context_ptr) {
  AS_LOG_S << "Creating pubkey; context " << context_ptr << std::endl;
  static_cast<aluminum_shark_Context*>(context_ptr)->context->createPublicKey();
  AS_LOG("Created pubkey");
}
void aluminum_shark_CreatePrivateKey(void* context_ptr) {
  AS_LOG_S << "Creating private key; context " << context_ptr << std::endl;
  static_cast<aluminum_shark_Context*>(context_ptr)
      ->context->createPrivateKey();
  AS_LOG("Created private key");
}

void aluminum_shark_SavePublicKey(const char* file, void* context_ptr) {
  AS_LOG("SavePublicKey not implemented");
}
void aluminum_shark_SavePrivateKey(const char* file, void* context_ptr) {
  AS_LOG("SavePrivateKey not implemented");
}
void aluminum_shark_LoadPublicKey(const char* file, void* context_ptr) {
  AS_LOG("LoadPublicKey not implemented");
}
void aluminum_shark_LoadPrivateKey(const char* file, void* context_ptr) {
  AS_LOG("LoadPrivateKey not implemented");
}

size_t aluminum_shark_numberOfSlots(void* context_ptr) {
  AS_LOG_S << "Calling number of slots on context: " << context_ptr
           << std::endl;
  aluminum_shark_Context* context =
      static_cast<aluminum_shark_Context*>(context_ptr);
  return context->context->numberOfSlots();
}

// destory a context
void aluminum_shark_DestroyContext(void* context_ptr) {
  AS_LOG_S << "Destroying context: " << context_ptr << std::endl;
  aluminum_shark_Context* c = static_cast<aluminum_shark_Context*>(context_ptr);
  context_map.erase(c->context.get());
  delete c;
}

// Ciphertext operations

// return (void*)aluminum_shark_Ctxt*
void* aluminum_shark_encryptLong(const long* values, int size, const char* name,
                                 void* context_ptr) {
  AS_LOG_S << "Encrypting Long. Context: " << context_ptr << std::endl;
  aluminum_shark_Context* context =
      static_cast<aluminum_shark_Context*>(context_ptr);
  std::vector<long> ptxt_vec(values, values + size);
  AS_LOG_S << "Encrypting Long. Values: [ ";
  if (ptxt_vec.size() < 10) {
    for (size_t i = 0; i < ptxt_vec.size(); ++i) {
      AS_LOG_SA << ptxt_vec[i] << ", ";
    }
  } else {
    for (size_t i = 0; i < 5; ++i) {
      AS_LOG_SA << ptxt_vec[i] << ", ";
    }
    AS_LOG_SA << " ... , ";
    for (size_t i = ptxt_vec.size() - 5; i < ptxt_vec.size(); ++i) {
      AS_LOG_SA << ptxt_vec[i] << ", ";
    }
  }
  AS_LOG_SA << " ] number of values (passed/read) " << size << "/"
            << ptxt_vec.size() << ", name: " << name << std::endl;

  std::shared_ptr<aluminum_shark::HECtxt> hectxt =
      std::shared_ptr<aluminum_shark::HECtxt>(
          context->context->encrypt(ptxt_vec, name));

  aluminum_shark_Ctxt* ret = new aluminum_shark_Ctxt();
  ret->ctxt = std::make_shared<aluminum_shark::Ctxt>(hectxt, name);
  AS_LOG_S << "new ctxt: " << reinterpret_cast<void*>(ret) << " wrapped object"
           << ret->ctxt << std::endl;
  return ret;
}

// return (void*)aluminum_shark_Ctxt*
void* aluminum_shark_encryptDouble(const double* values, int size,
                                   const char* name, void* context_ptr) {
  AS_LOG_S << "Encrypting Double. Context: " << context_ptr << std::endl;
  aluminum_shark_Context* context =
      static_cast<aluminum_shark_Context*>(context_ptr);
  std::vector<double> ptxt_vec(values, values + size);
  AS_LOG_S << "Encrypting Double. Values: [ ";
  if (ptxt_vec.size() < 10) {
    for (size_t i = 0; i < ptxt_vec.size(); ++i) {
      AS_LOG_SA << ptxt_vec[i] << ", ";
    }
  } else {
    for (size_t i = 0; i < 5; ++i) {
      AS_LOG_SA << ptxt_vec[i] << ", ";
    }
    AS_LOG_SA << " ... , ";
    for (size_t i = ptxt_vec.size() - 5; i < ptxt_vec.size(); ++i) {
      AS_LOG_SA << ptxt_vec[i] << ", ";
    }
  }
  AS_LOG_SA << " ] number of values (passed/read) " << size << "/"
            << ptxt_vec.size() << ", name: " << name << std::endl;
  std::shared_ptr<aluminum_shark::HECtxt> hectxt =
      std::shared_ptr<aluminum_shark::HECtxt>(
          context->context->encrypt(ptxt_vec, name));

  aluminum_shark_Ctxt* ret = new aluminum_shark_Ctxt();
  // create shared_prt with empty ctxt and then copy the result in
  ret->ctxt = std::make_shared<aluminum_shark::Ctxt>(hectxt, name);
  AS_LOG_S << "new ctxt: " << reinterpret_cast<void*>(ret) << " wrapped object"
           << ret->ctxt << std::endl;
  return ret;
}

void aluminum_shark_decryptLong(long* ret, int* ret_size, void* ctxt_ptr,
                                void* context_ptr) {
  AS_LOG_S << "Decrypt long ciphertext: " << ctxt_ptr << " context: "
           << " return array: " << reinterpret_cast<void*>(ret)
           << " return size addr: " << reinterpret_cast<void*>(ret_size)
           << std::endl;
  aluminum_shark_Ctxt* ctxt = static_cast<aluminum_shark_Ctxt*>(ctxt_ptr);
  aluminum_shark_Context* context =
      static_cast<aluminum_shark_Context*>(context_ptr);
  std::vector<long> vec =
      context->context->decryptLong(ctxt->ctxt->getValuePtr().get());
  AS_LOG_S << "Decrypted Long. Values: [ ";
  if (vec.size() < 10) {
    for (size_t i = 0; i < vec.size(); ++i) {
      AS_LOG_SA << vec[i] << ", ";
    }
  } else {
    for (size_t i = 0; i < 5; ++i) {
      AS_LOG_SA << vec[i] << ", ";
    }
    AS_LOG_SA << " ... , ";
    for (size_t i = vec.size() - 5; i < vec.size(); ++i) {
      AS_LOG_SA << vec[i] << ", ";
    }
  }
  AS_LOG_SA << "number of values: " << vec.size() << std::endl;
  *ret_size = vec.size();
  std::copy(vec.begin(), vec.end(), ret);
}

void aluminum_shark_decryptDouble(double* ret, int* ret_size, void* ctxt_ptr,
                                  void* context_ptr) {
  AS_LOG_S << "Decrypt double ciphertext: " << ctxt_ptr << " context: "
           << " return array: " << reinterpret_cast<void*>(ret)
           << " return size addr: " << reinterpret_cast<void*>(ret_size)
           << std::endl;
  aluminum_shark_Ctxt* ctxt = static_cast<aluminum_shark_Ctxt*>(ctxt_ptr);
  aluminum_shark_Context* context =
      static_cast<aluminum_shark_Context*>(context_ptr);
  std::vector<double> vec =
      context->context->decryptDouble(ctxt->ctxt->getValuePtr().get());
  AS_LOG_S << "Decrypted Double. Values: [ ";
  if (vec.size() < 10) {
    for (size_t i = 0; i < vec.size(); ++i) {
      AS_LOG_SA << vec[i] << ", ";
    }
  } else {
    for (size_t i = 0; i < 5; ++i) {
      AS_LOG_SA << vec[i] << ", ";
    }
    AS_LOG_SA << " ... , ";
    for (size_t i = vec.size() - 5; i < vec.size(); ++i) {
      AS_LOG_SA << vec[i] << ", ";
    }
  }
  AS_LOG_SA << "number of values: " << vec.size() << std::endl;
  *ret_size = vec.size();
  std::copy(vec.begin(), vec.end(), ret);
}

// destroy a ciphertext
void aluminum_shark_DestroyCiphertext(void* ctxt_ptr) {
  AS_LOG_S << "destroying ciphertext: " << ctxt_ptr << std::endl;
  aluminum_shark_Ctxt* ptr = static_cast<aluminum_shark_Ctxt*>(ctxt_ptr);
  AS_LOG("destroying ciphertext: " + ptr->ctxt->getName());
  delete static_cast<aluminum_shark_Ctxt*>(ctxt_ptr);
}

void* aluminum_shark_RegisterComputation(void* (*ctxt_callback)(int*),
                                         void (*result_callback)(void*, int)) {
  aluminum_shark_Computation* ret = new aluminum_shark_Computation();
  ret->computation = std::make_shared<aluminum_shark::ComputationHandle>(
      ctxt_callback, result_callback);
  auto& pyh = ::aluminum_shark::PythonHandle::getInstance();
  pyh.registerComputation(ret->computation);
  return ret;
}

// // set the ciphertexts used for the next computation
// void aluminum_shark_SetChipherTexts(void* values, const int size) {
//   AS_LOG_S << "setting ciphertext: " << values << "number:" << size
//            << std::endl;
//   aluminum_shark_Ctxt** ctxts = static_cast<aluminum_shark_Ctxt**>(values);
//   std::vector<aluminum_shark::Ctxt> currentCtxts;
//   AS_LOG_S << "Ctxts: [ ";
//   for (int i = 0; i < size; ++i) {
//     AS_LOG_SA << reinterpret_cast<void*>(ctxts[i]) << ", ";
//     aluminum_shark::Ctxt ctxt = ctxts[i]->ctxt.operator*();
//     currentCtxts.push_back(ctxt);
//   }
//   AS_LOG_SA << "] " << std::endl;
//   auto& pyh = aluminum_shark::PythonHandle::getInstance();
//   pyh.setCiphertexts(currentCtxts);
// }

// void* aluminum_shark_GetChipherTextResult(void** context_return) {
//   auto& pyh = aluminum_shark::PythonHandle::getInstance();
//   const aluminum_shark::Ctxt ctxt = pyh.retriveCiphertextsResults();
//   aluminum_shark_Ctxt* ret = new aluminum_shark_Ctxt();
//   // create shared_prt with a copy of the result
//   ret->ctxt = std::make_shared<aluminum_shark::Ctxt>(ctxt);
//   AS_LOG_S << "Getting ciphertext result: " << reinterpret_cast<void*>(ret)
//            << " wrapped object:" << ret->ctxt << std::endl;
//   auto it = context_map.find(ctxt.getValue().getContext());
//   if (it == context_map.end()) {
//     AS_LOG_S << "couldnt find the context" << std::endl;
//   }
//   AS_LOG_S << "ret" << reinterpret_cast<void*>(context_return) << std::endl;
//   *context_return = it->second;
//   AS_LOG_S << "ret " << reinterpret_cast<void*>(context_return) << std::endl;
//   AS_LOG_S << "*ret " << reinterpret_cast<void*>(*context_return) <<
//   std::endl; return ret;
// }

}  // extern "C"