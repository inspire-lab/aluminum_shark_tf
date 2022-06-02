#include "tensorflow/compiler/plugin/aluminum_shark/ptxt.h"

#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

namespace aluminum_shark {

// constructors

Ptxt::Ptxt(std::string name) : name_(name) {}

Ptxt::Ptxt(std::vector<std::shared_ptr<HEPtxt>> heptxt,
           std::shared_ptr<Layout> layout, std::string name)
    : BaseTxt(layout), value_(heptxt), name_(name) {}

// create a deep copy which also creates a copy of stored object
Ptxt Ptxt::deepCopy() const {
  AS_LOG("creating deep copy of: " + name_);
  Ptxt copy = *this;
  // create a copy of the stored object
  std::vector<std::shared_ptr<HEPtxt>> heptxt_copy;
  for (auto heptxt : value_) {
    heptxt_copy.push_back(std::shared_ptr<HEPtxt>(heptxt->deepCopy()));
  }
  copy.setValue(heptxt_copy);
  // copy layout
  copy.setLayout(std::shared_ptr<Layout>(copy.layout().deepCopy()));
  return copy;
  return copy;
}

bool Ptxt::is_initialized() const { return value_.size() != 0; }

// getters and setters

const std::vector<std::shared_ptr<HEPtxt>>& Ptxt::getValue() const {
  return value_;
}

std::vector<std::shared_ptr<HEPtxt>>& Ptxt::getValue() { return value_; }

const std::string& Ptxt::getName() const { return name_; }
void Ptxt::setName(const std::string& name) { name_ = name; }

void Ptxt::setValue(std::vector<std::shared_ptr<HEPtxt>>& value_ptrs) {
  AS_LOG_S << "setting HEPtxts for " << name_ << std::endl;
  value_ = value_ptrs;
}

// BaseTxt interface

std::string Ptxt::to_string() const {
  if (is_initialized()) {
    return "Plaintext is not initialized.";
  }
  // TODO: get more info
  return name_;
}

void Ptxt::updateLayout(std::shared_ptr<Layout> layout) {
  // TODO:
}

std::shared_ptr<BaseTxt> Ptxt::operator+(const BaseTxt& other) const {
  // need to handle the ctxt seperatly. we can't use inplace operations here
  // since the result will be a ctxt
  try {
    const Ctxt& ctxt = dynamic_cast<const Ctxt&>(other);
    return ctxt + *this;
  } catch (const std::bad_cast& e) {
    /* this is ok. ignore */
  }
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr += other;
  return ptr;
}

std::shared_ptr<BaseTxt> Ptxt::operator*(const BaseTxt& other) const {
  // need to handle the ctxt seperatly. we can't use inplace operations here
  // since the result will be a ctxt
  try {
    const Ctxt& ctxt = dynamic_cast<const Ctxt&>(other);
    return ctxt * *this;
  } catch (const std::bad_cast& e) {
    /* this is ok. ignore */
  }
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr *= other;
  return ptr;
}

BaseTxt& Ptxt::operator+=(const BaseTxt& other) {
  // if we made this far this needs to be a ptxt
  try {
    const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
    layout().add_in_place(*this, ptxt);
    name_ += " + " + ptxt.getName();
    return *this;
  } catch (const std::bad_cast& e) {
    // this is not ok
    AS_LOG_S << e.what() << std::endl;
    throw e;
  }
}

BaseTxt& Ptxt::operator*=(const BaseTxt& other) {
  // if we made this far this needs to be a ptxt
  try {
    const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
    layout().multiply_in_place(*this, ptxt);
    name_ += " + " + ptxt.getName();
    return *this;
  } catch (const std::bad_cast& e) {
    // this is not ok
    AS_LOG_S << e.what() << std::endl;
    throw e;
  }
}

// integer
std::shared_ptr<BaseTxt> Ptxt::operator+(long other) {
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr += other;
  return ptr;
}

std::shared_ptr<BaseTxt> Ptxt::operator*(long other) {
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr += other;
  return ptr;
}

Ptxt& Ptxt::operator+=(long other) {
  layout().add_in_place(*this, other);
  name_ += " + " + std::to_string(other);
  return *this;
}

Ptxt& Ptxt::operator*=(long other) {
  layout().multiply_in_place(*this, other);
  name_ += " * " + std::to_string(other);
  return *this;
}

// floating point
std::shared_ptr<BaseTxt> Ptxt::operator+(double other) {
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr += other;
  return ptr;
}

std::shared_ptr<BaseTxt> Ptxt::operator*(double other) {
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr *= other;
  return ptr;
}

Ptxt& Ptxt::operator+=(double other) {
  layout().add_in_place(*this, other);
  name_ += " + " + std::to_string(other);
  return *this;
}

Ptxt& Ptxt::operator*=(double other) {
  layout().multiply_in_place(*this, other);
  name_ += " * " + std::to_string(other);
  return *this;
}

// TODO RP: template this
std::vector<double> Ptxt::decodeDouble() const {
  std::vector<std::vector<double>> decodings;
  for (const auto& heptxt : value_) {
    decodings.push_back(heptxt->getContext()->decodeDouble(heptxt.get()));
  }
  std::vector<double> vec = layout().reverse_layout_vector(decodings);
  AS_LOG_S << "Decoded Double. Values: [ ";
  if (log()) {
    aluminum_shark::stream_vector(vec);
  }
  AS_LOG_SA << "number of values: " << vec.size() << std::endl;
  return vec;
}

// TODO RP: template this
std::vector<long> Ptxt::decodeLong() const {
  std::vector<std::vector<long>> decodings;
  for (const auto& heptxt : value_) {
    decodings.push_back(heptxt->getContext()->decodeLong(heptxt.get()));
  }
  std::vector<long> vec = layout().reverse_layout_vector(decodings);
  AS_LOG_S << "Decoded long. Values: [ ";
  if (log()) {
    aluminum_shark::stream_vector(vec);
  }
  AS_LOG_SA << "number of values: " << vec.size() << std::endl;
  return vec;
}

}  // namespace aluminum_shark
