#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PTXT_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PTXT_H

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/base_txt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/he_backend/he_backend.h"

namespace aluminum_shark {

// A class that wraps around a storage object which holds the actual data.
// Needs to be cheap to copy which is why the storage object is held in a shared
// pointer. But this means none of these objects actually exclusively own the
// stored object. this class holds an encoded HEPtxt
//
class Ptxt : public BaseTxt {
 public:
  // Ctors
  Ptxt() {}
  Ptxt(std::string name) : name_(name) {}
  Ptxt(std::shared_ptr<HEPtxt> ptxt, std::string name);
  // Takes over the owernship of ptxt_ptr
  Ptxt(HEPtxt* ptxt_ptr, std::string name)
      : Ptxt(std::shared_ptr<HEPtxt>(ptxt_ptr), name) {}

  virtual ~Ptxt(){};

  // setters and getters
  void setValue(std::shared_ptr<HEPtxt> value_ptr);
  const HEPtxt& getValue() const;
  HEPtxt& getValue();
  std::shared_ptr<HEPtxt> getValuePtr() const;

  void setName(const std::string& name);
  const std::string& getName() const;

  // create a deep copy which also creates a copy of the stored object
  Ptxt deepCopy() const;

  std::string to_string() const override;

  // BaseTxt interface
  std::shared_ptr<BaseTxt> operator+(const BaseTxt& other) const override;
  std::shared_ptr<BaseTxt> operator*(const BaseTxt& other) const override;

  BaseTxt& operator+=(const BaseTxt& other) override;
  BaseTxt& operator*=(const BaseTxt& other) override;

  // integer
  std::shared_ptr<BaseTxt> operator+(long other) override;
  std::shared_ptr<BaseTxt> operator*(long other) override;
  Ptxt& operator+=(long other) override;
  Ptxt& operator*=(long other) override;
  // floating point
  std::shared_ptr<BaseTxt> operator+(double other) override;
  std::shared_ptr<BaseTxt> operator*(double other) override;
  Ptxt& operator+=(double other) override;
  Ptxt& operator*=(double other) override;

 private:
  std::shared_ptr<HEPtxt> value_;
  std::string name_;
};

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PTXT_H \
        */
