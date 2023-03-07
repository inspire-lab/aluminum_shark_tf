#include "tensorflow/compiler/plugin/aluminum_shark/python/python_handle.h"

#include <cstring>
#include <iostream>
#include <map>
#include <memory>

#include "tensorflow/compiler/plugin/aluminum_shark/python/arg_utils.h"
#include "tensorflow/compiler/plugin/aluminum_shark/utils/parallel.h"
#include "tensorflow/compiler/plugin/aluminum_shark/utils/utils.h"
// #include <mutex>

namespace {
static aluminum_shark::Ctxt I_AM_ERROR("I AM ERROR");

const bool parallel =
    std::getenv("ALUMINUM_SHARK_PARALLEL_ENCRYPTION") == nullptr
        ? false
        : std::stoi(std::getenv("ALUMINUM_SHARK_PARALLEL_ENCRYPTION")) == 1;

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
    if (clear_memory_) {
      delete as_ctxt;
    }
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

void ComputationHandle::start_operation_callback(const std::string& name) {
  monitor_progress_callback_(name.c_str(), true);
}
void ComputationHandle::end_operation_callback(const std::string& name) {
  monitor_progress_callback_(name.c_str(), false);
}

void ComputationHandle::log_value_callback(const std::string& name,
                                           double value) {
  monitor_value_callback_(name.c_str(), value);
}

bool ComputationHandle::useForcedLayout() const {
  return forced_layout_ != nullptr;
};

const char* ComputationHandle::getForcedLayout() const {
  return forced_layout_;
}

bool ComputationHandle::clearMemory() const { return clear_memory_; }

}  // namespace aluminum_shark

/***********************/
/*  C API              */
/***********************/

using ::aluminum_shark::operator<<;

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

// turns the ressource monitor on or off for this backend
void aluminum_shark_enable_ressource_monitor(bool enable, void* backend_ptr) {
  static_cast<aluminum_shark_HEBackend*>(backend_ptr)
      ->backend->enable_ressource_monitor(enable);
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

void* aluminum_shark_CreateContextCKKS_dynamic(aluminum_shark_Argument* args,
                                               int count, void* backend_ptr) {
  AS_LOG_INFO << "creating CKKS backend" << std::endl;
  std::vector<aluminum_shark_Argument> arguments;
  aluminum_shark_Argument** arrrrgs =
      reinterpret_cast<aluminum_shark_Argument**>(args);
  for (size_t i = 0; i < count; i++) {
    AS_LOG_INFO << aluminum_shark::arg_to_string(*arrrrgs[i]) << std::endl;
    arguments.push_back(*arrrrgs[i]);
  }

  auto backend = static_cast<aluminum_shark_HEBackend*>(backend_ptr)->backend;
  aluminum_shark_Context* ret = new aluminum_shark_Context();
  ret->context = std::shared_ptr<aluminum_shark::HEContext>(
      backend->createContextCKKS(arguments));
  context_map[ret->context.get()] = ret;
  return ret;
}

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

// Layout stuff

const char* const* aluminum_shark_GetAvailabeLayouts(size_t* size) {
  *size = aluminum_shark::LAYOUT_TYPE_C_STRINGS.size();
  return aluminum_shark::LAYOUT_TYPE_C_STRINGS.data();
}

// Ciphertext operations

// return (void*)aluminum_shark_Ctxt*
void* aluminum_shark_encryptLong(const long* values, int size, const char* name,
                                 const size_t* shape, int shape_size,
                                 const char* layout, void* context_ptr) {
  AS_LOG_S << "Encrypting Long. Context: " << context_ptr << std::endl;
  aluminum_shark_Context* context =
      static_cast<aluminum_shark_Context*>(context_ptr);
  std::vector<long> ptxt_vec(values, values + size);
  AS_LOG_S << "Encrypting Long. Values: [ ";
  if (aluminum_shark::log()) {
    aluminum_shark::stream_vector(ptxt_vec);
  }
  AS_LOG_SA << " ] number of values (passed/read) " << size << "/"
            << ptxt_vec.size() << ", name: " << name << std::endl;

  aluminum_shark::Shape shape_v(shape, shape + shape_size);
  aluminum_shark::Layout* layout_ =
      aluminum_shark::createLayout(layout, shape_v);
  auto layedout_vec = layout_->layout_vector(ptxt_vec);
  std::vector<shared_ptr<aluminum_shark::HECtxt>> ctxts;
  for (auto& v : layedout_vec) {
    ctxts.push_back(
        shared_ptr<aluminum_shark::HECtxt>(context->context->encrypt(v, name)));
  }

  aluminum_shark_Ctxt* ret = new aluminum_shark_Ctxt();
  ret->ctxt = std::make_shared<aluminum_shark::Ctxt>(
      ctxts, std::shared_ptr<aluminum_shark::Layout>(layout_), name);
  AS_LOG_S << "new ctxt: " << reinterpret_cast<void*>(ret) << " wrapped object"
           << ret->ctxt << std::endl;
  return ret;
}

// return (void*)aluminum_shark_Ctxt*
void* aluminum_shark_encryptDouble(const double* values, int size,
                                   const char* name, const size_t* shape,
                                   int shape_size, const char* layout_type,
                                   void* context_ptr) {
  AS_LOG_INFO << "Encrypting Double. Context: " << context_ptr << std::endl;
  aluminum_shark_Context* context =
      static_cast<aluminum_shark_Context*>(context_ptr);
  // read input values
  std::vector<double> ptxt_vec(values, values + size);
  if (aluminum_shark::log_large_vectors()) {
    AS_LOG_DEBUG << "Encrypting Double. Values: " << ptxt_vec << std::endl;
  }
  AS_LOG_INFO << "number of values (passed/read) " << size << "/"
              << ptxt_vec.size() << ", name: " << name << std::endl;

  // read shape and create layout
  AS_LOG_INFO << "Creating layout, shape: " << shape << ", " << shape_size
              << std::endl;
  std::vector<size_t> shape_vec(shape, shape + shape_size);

  AS_LOG_S << "layout: " << layout_type << std::endl;
  aluminum_shark::Layout* layout =
      aluminum_shark::createLayout(layout_type, shape_vec);
  // layout ptxt and encrypt
  AS_LOG_S << "Layout created" << std::endl;
  AS_LOG_S << layout->type() << std::endl;
  std::vector<std::vector<double>> ptxt_with_layout =
      layout->layout_vector(ptxt_vec);

  AS_LOG_S << "Input layed out" << std::endl;
  std::vector<shared_ptr<aluminum_shark::HECtxt>> hectxts;
  if (parallel) {
    // running multithreaded encryption
    hectxts = std::vector<shared_ptr<aluminum_shark::HECtxt>>(
        ptxt_with_layout.size());
    auto func = [&ptxt_with_layout, &hectxts, &context, &name](size_t i) {
      hectxts[i] = shared_ptr<aluminum_shark::HECtxt>(
          context->context->encrypt(ptxt_with_layout[i], name));
    };
    aluminum_shark::run_parallel(0, ptxt_with_layout.size(), func);
  } else {
    for (auto& v : ptxt_with_layout) {
      hectxts.push_back(shared_ptr<aluminum_shark::HECtxt>(
          context->context->encrypt(v, name)));
    }
  }
  if (hectxts.size() > 0) {
    AS_LOG_INFO << "Ciphertext size: " << hectxts[0]->size() << " bytes"
                << std::endl;
  }
  AS_LOG_DEBUG << "Input encrypted" << std::endl;
  // create ctxt and wrap it for python
  aluminum_shark_Ctxt* ret = new aluminum_shark_Ctxt();
  // create shared_prt with empty ctxt and then copy the result in
  ret->ctxt = std::make_shared<aluminum_shark::Ctxt>(
      hectxts, std::shared_ptr<aluminum_shark::Layout>(layout), name);
  AS_LOG_S << "new ctxt: " << reinterpret_cast<void*>(ret) << " wrapped object "
           << ret->ctxt << std::endl;
  return ret;
}

void aluminum_shark_decryptLong(long* ret, void* ctxt_ptr, void* context_ptr) {
  AS_LOG_S << "Decrypt long ciphertext: " << ctxt_ptr << " context: "
           << " return array: " << reinterpret_cast<void*>(ret) << std::endl;
  aluminum_shark_Ctxt* ctxt = static_cast<aluminum_shark_Ctxt*>(ctxt_ptr);
  aluminum_shark_Context* context =
      static_cast<aluminum_shark_Context*>(context_ptr);
  std::vector<long> vec;
  try {
    vec = ctxt->ctxt->decryptLong();
  } catch (const std::exception& e) {
    AS_LOG_S << e.what() << std::endl;
  }
  std::copy(vec.begin(), vec.end(), ret);
}

void aluminum_shark_decryptDouble(double* ret, void* ctxt_ptr,
                                  void* context_ptr) {
  AS_LOG_S << "Decrypt double ciphertext: " << ctxt_ptr << " context: "
           << " return array: " << reinterpret_cast<void*>(ret) << std::endl;
  aluminum_shark_Ctxt* ctxt = static_cast<aluminum_shark_Ctxt*>(ctxt_ptr);
  aluminum_shark_Context* context =
      static_cast<aluminum_shark_Context*>(context_ptr);
  std::vector<double> vec;
  try {
    vec = ctxt->ctxt->decryptDouble();
  } catch (const std::exception& e) {
    AS_LOG_S << e.what() << std::endl;
  }

  std::copy(vec.begin(), vec.end(), ret);
}

size_t aluminum_shark_GetCtxtShapeLen(void* ctxt_ptr) {
  aluminum_shark_Ctxt* ctxt = reinterpret_cast<aluminum_shark_Ctxt*>(ctxt_ptr);
  return ctxt->ctxt->shape().size();
}

void aluminum_shark_GetCtxtShape(void* ctxt_ptr, size_t* shape_array) {
  aluminum_shark_Ctxt* ctxt = reinterpret_cast<aluminum_shark_Ctxt*>(ctxt_ptr);
  for (size_t i = 0; i < ctxt->ctxt->shape().size(); ++i) {
    shape_array[i] = ctxt->ctxt->shape()[i];
  }
}

// destroy a ciphertext
void aluminum_shark_DestroyCiphertext(void* ctxt_ptr) {
  AS_LOG_S << "destroying ciphertext: " << ctxt_ptr << std::endl;
  aluminum_shark_Ctxt* ptr = static_cast<aluminum_shark_Ctxt*>(ctxt_ptr);
  AS_LOG("destroying ciphertext: " + ptr->ctxt->getName());
  delete static_cast<aluminum_shark_Ctxt*>(ctxt_ptr);
}

void* aluminum_shark_RegisterComputation(
    void* (*ctxt_callback)(int*), void (*result_callback)(void*, int),
    void (*monitor_value_callback)(const char*, double),
    void (*monitor_progress_callback)(const char*, bool),
    const char* forced_layout, bool clear_memory) {
  aluminum_shark_Computation* ret = new aluminum_shark_Computation();
  if (forced_layout && strlen(forced_layout) == 0) {
    ret->computation = std::make_shared<aluminum_shark::ComputationHandle>(
        ctxt_callback, result_callback, monitor_value_callback,
        monitor_progress_callback, nullptr, clear_memory);
  } else {
    ret->computation = std::make_shared<aluminum_shark::ComputationHandle>(
        ctxt_callback, result_callback, monitor_value_callback,
        monitor_progress_callback, forced_layout, clear_memory);
  }
  auto& pyh = ::aluminum_shark::PythonHandle::getInstance();
  pyh.registerComputation(ret->computation);
  return ret;
}

// turns logging on or off
void aluminum_shark_EnableLogging(bool on) {
  aluminum_shark::enable_logging(on);
}

// sets the log level
void aluminum_shark_SetLogLevel(int level) {
  aluminum_shark::set_log_level(level);
}

// sets the log level
void aluminum_shark_SetBackendLogLevel(int level, void* backend_ptr) {
  std::shared_ptr<aluminum_shark::HEBackend> backend =
      static_cast<aluminum_shark_HEBackend*>(backend_ptr)->backend;
  backend->set_log_level(level);
}

}  // extern "C"
