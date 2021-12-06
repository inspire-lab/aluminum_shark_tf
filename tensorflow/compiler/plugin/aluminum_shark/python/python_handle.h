#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_PYTHON_HANDLE_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_PYTHON_HANDLE_H

#include <list>
#include <map>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/dummy_data_type.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

/*
 * Singleton Class that keeps track of c++ objects accessed via python.
 */

namespace aluminum_shark {

class PythonHandle {
 public:
  static PythonHandle& getInstance() {
    AS_LOG("getting PythonHandle instance");
    static PythonHandle instance;
    return instance;
  }

  // Python facing methods
  // creates ciphtertexts from the inputs
  template <typename T>
  DummyDataType* encrypt(const std::vector<T>& ptxts, const std::string& name) {
    AS_LOG("encrypting " + name);
    DummyDataType ddt;
    std::vector<long> vec(ptxts.size(), 0);
    for (size_t i = 0; i < vec.size(); ++i) {
      vec[i] = static_cast<long>(ptxts[i]);
    }
    ddt.setValue(std::move(vec));
    ddt.setName(name);
    registry_.push_back(std::move(ddt));
    return &registry_.back();
  }

  // decrypts the ciphtertexts inputs
  void decrypt(std::vector<long>* ret, const DummyDataType& ddt);
  void decrypt(std::vector<long>* ret, const std::string& name);

  // set the ciphertext inputs for the current computation
  void setCiphertexts(DummyDataType& ddt);
  void setCiphertexts(const std::string& name);

  // retrive the current computation ciphertexts results
  DummyDataType* retriveCiphertextsResults();
  // const std::string& retriveCiphertextsResults();

  // C++ facing methods
  const DummyDataType& getCurrentCiphertexts();

  void setCurrentResult(DummyDataType& ddt);

  PythonHandle(PythonHandle const&) = delete;
  void operator=(PythonHandle const&) = delete;

 private:
  PythonHandle() {}

  DummyDataType& find(const std::string& name);

  DummyDataType* input_;
  DummyDataType* result_;

  std::list<DummyDataType> registry_;
};

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_PYTHON_HANDLE_H \
        */
