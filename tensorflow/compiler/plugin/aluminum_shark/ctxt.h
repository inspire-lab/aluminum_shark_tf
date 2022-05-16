#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_CTXT_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_CTXT_H

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/base_txt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/he_backend/he_backend.h"
#include "tensorflow/compiler/plugin/aluminum_shark/ptxt.h"

namespace aluminum_shark {

// A class that wraps around a storage object which holds the actual data.
// Needs to be cheap to copy which is why the storage object is held in a shared
// pointer. But this means none of these objects actually exclusively own the
// stored object.
//
class Ctxt : public BaseTxt {
 public:
  // Ctor/Dtor
  Ctxt() = default;
  //   Ctxt(const Ctxt& other) = default;
  //   Ctxt(Ctxt&& other) = default;
  Ctxt(std::string name);
  Ctxt(std::vector<std::shared_ptr<HECtxt>> hectxt,
       std::shared_ptr<Layout> layout, std::string name);
  virtual ~Ctxt() {}

  // getters / setters
  const std::vector<std::shared_ptr<HECtxt>>& getValue() const;
  std::vector<std::shared_ptr<HECtxt>>& getValue();
  void setValue(std::vector<std::shared_ptr<HECtxt>>& value_ptrs);

  const std::string& getName() const;
  void setName(const std::string& name);

  // create a deep copy which also creates a copy of the stored object
  Ctxt deepCopy() const;

  const HEContext* getContext() const;

  // BaseTxttxt interface
  std::string to_string() const override;

  void updateLayout(std::shared_ptr<Layout> layout) override;

  std::shared_ptr<BaseTxt> operator+(const BaseTxt& other) const;
  std::shared_ptr<BaseTxt> operator*(const BaseTxt& other) const;

  Ctxt& operator+=(const BaseTxt& other);
  Ctxt& operator*=(const BaseTxt& other);

  // integer
  std::shared_ptr<BaseTxt> operator+(long other);
  std::shared_ptr<BaseTxt> operator*(long other);
  Ctxt& operator+=(long other);
  Ctxt& operator*=(long other);
  // floating point
  std::shared_ptr<BaseTxt> operator+(double other);
  std::shared_ptr<BaseTxt> operator*(double other);
  Ctxt& operator+=(double other);
  Ctxt& operator*=(double other);

 private:
  std::vector<std::shared_ptr<HECtxt>> value_;
  std::string name_;
};

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_CTXT_H \
        */
