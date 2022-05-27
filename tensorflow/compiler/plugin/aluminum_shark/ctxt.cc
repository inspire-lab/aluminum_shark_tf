#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"

#include <cxxabi.h>

#include <typeinfo>

#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

namespace aluminum_shark {

// constructors

Ctxt::Ctxt(std::string name) : name_(name) {}

Ctxt::Ctxt(std::vector<std::shared_ptr<HECtxt>> hectxt,
           std::shared_ptr<Layout> layout, std::string name)
    : BaseTxt(layout), value_(hectxt), name_(name) {}

// create a deep copy which also creates a copy of stored object
Ctxt Ctxt::deepCopy() const {
  AS_LOG("creating deep copy of: " + name_);
  Ctxt copy = *this;
  // create a copy of the stored object
  std::vector<std::shared_ptr<HECtxt>> hectxt_copy;
  for (auto hectxt : value_) {
    hectxt_copy.push_back(std::shared_ptr<HECtxt>(hectxt->deepCopy()));
  }
  copy.setValue(hectxt_copy);
  // copy layout
  copy.setLayout(std::shared_ptr<Layout>(copy.layout().deepCopy()));
  return copy;
}

const HEContext* Ctxt::getContext() const {
  if (value_.size() == 0) {
    return nullptr;
  }
  return value_[0]->getContext();
}

// getters and setters

const std::vector<std::shared_ptr<HECtxt>>& Ctxt::getValue() const {
  return value_;
}

std::vector<std::shared_ptr<HECtxt>>& Ctxt::getValue() { return value_; }

void Ctxt::setValue(std::vector<std::shared_ptr<HECtxt>>& value_ptrs) {
  AS_LOG_S << "setting HECtxts for " << name_ << std::endl;
  value_ = value_ptrs;
}

const std::string& Ctxt::getName() const { return name_; }
void Ctxt::setName(const std::string& name) { name_ = name; }

// BaseTxttxt interface

std::string Ctxt::to_string() const {
  if (!value_.size() == 0) {
    return "Ciphertext is not initialized.";
  }
  // TODO: get more info
  return name_;
}

void Ctxt::updateLayout(std::shared_ptr<Layout> layout) {
  // TODO:
}

std::shared_ptr<BaseTxt> Ctxt::operator+(const BaseTxt& other) const {
  // create a copy
  AS_LOG_S << "adding to ciphertext" << std::endl;
  auto ptr = std::make_shared<Ctxt>(deepCopy());
  // call in place operator
  *ptr += other;
  return ptr;
}
std::shared_ptr<BaseTxt> Ctxt::operator*(const BaseTxt& other) const {
  // create a copy
  auto ptr = std::make_shared<Ctxt>(deepCopy());
  // call in place operator
  *ptr *= other;
  return ptr;
}

Ctxt& Ctxt::operator+=(const BaseTxt& other) {
  AS_LOG_S << "inplace add" << std::endl;
  try {
    const Ctxt& ctxt = dynamic_cast<const Ctxt&>(other);
    AS_LOG_S << "adding ciphertext" << std::endl;
    layout().add_in_place(*this, ctxt);
    name_ += " + " + ctxt.getName();
    return *this;
  } catch (const std::bad_cast& e) {
    /* this is ok. ignore */
  }
  try {
    const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
    AS_LOG_S << "adding plaintext" << std::endl;
    AS_LOG_S << "layout " << layout().type() << std::endl;
    layout().add_in_place(*this, ptxt);
    name_ += " + " + ptxt.getName();
    AS_LOG_S << "new name: " << name_ << std::endl;
    return *this;
  } catch (const std::bad_cast& e) {
    // this is not ok
    AS_LOG_S << e.what() << std::endl;
    throw e;
  }
}

Ctxt& Ctxt::operator*=(const BaseTxt& other) {
  try {
    const Ctxt& ctxt = dynamic_cast<const Ctxt&>(other);
    AS_LOG_S << "Calling in place multipication: " << name_
             << " *= " << ctxt.getName() << std::endl;
    AS_LOG_S << "using " << layout_type_to_string(layout().type()) << " layout"
             << std::endl;
    layout().multiply_in_place(*this, ctxt);
    name_ += " * " + ctxt.getName();
    return *this;
  } catch (const std::bad_cast& e) {
    /* this is ok. ignore */
  }
  try {
    const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
    layout().multiply_in_place(*this, ptxt);
    name_ += " * " + ptxt.getName();
    return *this;
  } catch (const std::bad_cast& e) {
    // this is not ok
    AS_LOG_S << e.what() << std::endl;
    throw e;
  }
}

// integer
std::shared_ptr<BaseTxt> Ctxt::operator+(long other) {
  // create a copy
  auto ptr = std::make_shared<Ctxt>(deepCopy());
  // call in place operator
  *ptr += other;
  return ptr;
}

std::shared_ptr<BaseTxt> Ctxt::operator*(long other) {
  // create a copy
  auto ptr = std::make_shared<Ctxt>(deepCopy());
  // call in place operator
  *ptr *= other;
  return ptr;
}

Ctxt& Ctxt::operator+=(long other) {
  layout().add_in_place(*this, other);
  name_ += " + " + std::to_string(other);
  return *this;
}

Ctxt& Ctxt::operator*=(long other) {
  layout().multiply_in_place(*this, other);
  name_ += " * " + std::to_string(other);
  return *this;
}

// floating point
std::shared_ptr<BaseTxt> Ctxt::operator+(double other) {
  // create a copy
  auto ptr = std::make_shared<Ctxt>(deepCopy());
  // call in place operator
  *ptr += other;
  return ptr;
}

std::shared_ptr<BaseTxt> Ctxt::operator*(double other) {
  // create a copy
  auto ptr = std::make_shared<Ctxt>(deepCopy());
  // call in place operator
  *ptr *= other;
  return ptr;
}

Ctxt& Ctxt::operator+=(double other) {
  layout().add_in_place(*this, other);
  name_ += " + " + std::to_string(other);
  return *this;
}

Ctxt& Ctxt::operator*=(double other) {
  layout().multiply_in_place(*this, other);
  name_ += " * " + std::to_string(other);
  return *this;
}

// TODO RP: template this
std::vector<double> Ctxt::decryptDouble() const {
  std::vector<std::vector<double>> decryptions;
  for (const auto& hectxt : value_) {
    decryptions.push_back(getContext()->decryptDouble(hectxt.get()));
  }
  std::vector<double> vec = layout().reverse_layout_vector(decryptions);
  AS_LOG_S << "Decrypted Double. Values: [ ";
  aluminum_shark::stream_vector(vec);
  AS_LOG_SA << "number of values: " << vec.size() << std::endl;
  return vec;
}

std::vector<long> Ctxt::decryptLong() const {
  std::vector<std::vector<long>> decryptions;
  for (const auto& hectxt : value_) {
    decryptions.push_back(getContext()->decryptLong(hectxt.get()));
  }
  std::vector<long> vec = layout().reverse_layout_vector(decryptions);
  AS_LOG_S << "Decrypted Long. Values: [ ";
  aluminum_shark::stream_vector(vec);
  AS_LOG_SA << "number of values: " << vec.size() << std::endl;
  return vec;
}

}  // namespace aluminum_shark
