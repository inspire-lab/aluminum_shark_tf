#include "tensorflow/compiler/plugin/aluminum_shark/python/python_handle.h"

namespace {
static aluminum_shark::DummyDataType I_AM_ERROR({1}, "I AM ERROR");
}

namespace aluminum_shark {

// static PythonHandle& PythonHandle::getInstance() {
//   AS_LOG("getting PythonHandle instance");
//   static PythonHandle instance;
//   return instance;
// }

DummyDataType& PythonHandle::find(const std::string& name) {
  for (auto& one : registry_) {
    if (one.getName().compare(name) == 0) {
      return one;
    }
  }
  AS_LOG("Didn't find: " + name);
  return I_AM_ERROR;  // FIXME not ideal. Do better error handling
}

void PythonHandle::decrypt(std::vector<long>* ret, const DummyDataType& ddt) {
  AS_LOG("decrypting " + ddt.to_string());
  ret->clear();
  for (auto v : ddt.getValue()) {
    ret->push_back(v);
  }
}

void PythonHandle::decrypt(std::vector<long>* ret, const std::string& name) {
  decrypt(ret, find(name));
}

void PythonHandle::setCiphertexts(DummyDataType& ddt) {
  AS_LOG("setting ciphertext " + ddt.to_string());
  input_ = &ddt;
}

void PythonHandle::setCiphertexts(const std::string& name) {
  setCiphertexts(find(name));
}

DummyDataType* PythonHandle::retriveCiphertextsResults() {
  AS_LOG("retrieving result " + result_->to_string());
  return result_;
}

// const std::string& PythonHandle::retriveCiphertextsResults() {
//   return result_->getName();
// }

const DummyDataType& PythonHandle::getCurrentCiphertexts() {
  AS_LOG("getting current ciphertext " + input_->to_string());
  return *input_;
}

void PythonHandle::setCurrentResult(DummyDataType& ddt) {
  AS_LOG("seeting result " + ddt.to_string());
  result_ = &ddt;
}

}  // namespace aluminum_shark