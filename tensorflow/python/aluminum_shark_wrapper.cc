/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/plugin/aluminum_shark/python/python_handle.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
// #include "tensorflow/python/lib/core/safe_ptr.h"

namespace py = pybind11;
namespace as = aluminum_shark;

namespace aluminum_shark {

class Context;  // forward declartion for declaring friendship

class CipherTextHandle {
 public:
  CipherTextHandle() : ddt_(nullptr){};
  CipherTextHandle(DummyDataType* ddt) : ddt_(ddt){};

  DummyDataType* get() { return ddt_; };

 private:
  DummyDataType* ddt_;
  friend Context;
};

class Context {
 public:
  Context(){};

  CipherTextHandle encrypt(std::vector<long> ptxt,
                           const std::string name = "") {
    auto& pyHandle = as::PythonHandle::getInstance();
    DummyDataType* ddt = pyHandle.encrypt(ptxt, name);
    return CipherTextHandle(ddt);
  }

  std::vector<long> decrypt(CipherTextHandle& cth) {
    auto& pyHandle = as::PythonHandle::getInstance();
    std::vector<long> ret;
    pyHandle.decrypt(&ret, *cth.ddt_);
    return ret;
  }

  void createKeys() {}
};

}  // namespace aluminum_shark

PYBIND11_MODULE(_pywrap_aluminum_shark, m) {
  m.def(
      "setCiphertext",
      [](as::CipherTextHandle& handle) {
        auto& pyHandle = as::PythonHandle::getInstance();
        pyHandle.setCiphertexts(*handle.get());
      },
      "Sets the ciphertext as the input for the next computation",
      py::arg("ctxt"));

  m.def(
      "getCiphertext",
      [](const std::string& handle) {
        auto& pyHandle = as::PythonHandle::getInstance();
        as::DummyDataType* ddt = pyHandle.retriveCiphertextsResults();
        return as::CipherTextHandle(ddt);
      },
      "Gets the ciphertext with the specified handle", py::arg("handle"));

  py::class_<as::Context>(m, "ContextImpl")
      .def(py::init<>())
      .def("encrypt", &as::Context::encrypt,
           "Encrypts the input values using the context", py::arg("ptxt"),
           py::arg("name") = "")
      .def("decrypt", &as::Context::decrypt, "Decrypts the ciphertext",
           py::arg("ctxt"));

  py::class_<as::CipherTextHandle>(m, "CipherTextHandleImpl");
};
