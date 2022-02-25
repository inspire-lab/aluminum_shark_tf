#include "tensorflow/compiler/plugin/aluminum_shark/ptxt.h"

#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

namespace aluminum_shark {

Ptxt::Ptxt(std::shared_ptr<HEPtxt> heptxt, std::string name)
    : value_(heptxt), name_(name) {
  AS_LOG_S << "Created Ptxt " << name
           << " internal: " << reinterpret_cast<void*>(value_.get())
           << std::endl;
}

std::string Ptxt::to_string() const {
  if (!value_.get()) {
    return "Plaintext is not initialized.";
  }
  // TODO: get more info
  return name_;
}

bool Ptxt::is_initialized() const { return !!value_.get(); }

// create a deep copy which also creates a copy of stored object
Ptxt Ptxt::deepCopy() const {
  AS_LOG("creating deep copy of: " + name_);
  Ptxt copy = *this;
  // create a copy of the stored object
  std::shared_ptr<HEPtxt> temp_ptr(value_->deepCopy());
  copy.setValue(temp_ptr);
  return copy;
}

void Ptxt::setName(const std::string& name) { name_ = name; }

void Ptxt::setValue(std::shared_ptr<HEPtxt> value_ptr) {
  AS_LOG_S << "set HECtxt " << reinterpret_cast<void*>(value_ptr.get())
           << std::endl;
  value_ = value_ptr;
}
const HEPtxt& Ptxt::getValue() const { return *value_; }
HEPtxt& Ptxt::getValue() { return *value_; }
std::shared_ptr<HEPtxt> Ptxt::getValuePtr() const { return value_; }
const std::string& Ptxt::getName() const { return name_; }

/************************/
/* BaseTxt interface */
/************************/
std::shared_ptr<BaseTxt> Ptxt::operator+(const BaseTxt& other) const {
  try {
    const Ctxt& ctxt = dynamic_cast<const Ctxt&>(other);
    return std::make_shared<Ctxt>(*ctxt.getValuePtr() + value_.get(),
                                  name_ + " + " + ctxt.getName());
  } catch (const std::bad_cast& e) {
    /* this is ok. ignore */
  }
  const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
  return std::make_shared<Ptxt>(*value_ + ptxt.getValuePtr().get(),
                                name_ + " + " + ptxt.getName());
}

std::shared_ptr<BaseTxt> Ptxt::operator*(const BaseTxt& other) const {
  try {
    const Ctxt& ctxt = dynamic_cast<const Ctxt&>(other);
    return std::make_shared<Ctxt>(*ctxt.getValuePtr() * value_.get(),
                                  name_ + " * " + ctxt.getName());
  } catch (const std::bad_cast& e) {
    /* this is ok. ignore */
  }
  const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
  return std::make_shared<Ptxt>(*value_ + ptxt.getValuePtr().get(),
                                name_ + " * " + ptxt.getName());
}

BaseTxt& Ptxt::operator+=(const BaseTxt& other) {
  // can't ues += with ctxt. just fail if we try it.
  const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
  value_->addInPlace(ptxt.getValuePtr().get());
  return *this;
}

BaseTxt& Ptxt::operator*=(const BaseTxt& other) {
  // can't ues += with ctxt. just fail if we try it.
  const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
  value_->multInPlace(ptxt.getValuePtr().get());
  return *this;
}

// integer
std::shared_ptr<BaseTxt> Ptxt::operator+(long other) {
  return std::make_shared<Ptxt>(*value_ + other,
                                name_ + " + " + std::to_string(other));
}

std::shared_ptr<BaseTxt> Ptxt::operator*(long other) {
  return std::make_shared<Ptxt>(*value_ * other,
                                name_ + " * " + std::to_string(other));
}
Ptxt& Ptxt::operator+=(long other) {
  value_->addInPlace(other);
  return *this;
}

Ptxt& Ptxt::operator*=(long other) {
  value_->multInPlace(other);
  return *this;
}

// floating point
std::shared_ptr<BaseTxt> Ptxt::operator+(double other) {
  return std::make_shared<Ptxt>(*value_ + other,
                                name_ + " + " + std::to_string(other));
}

std::shared_ptr<BaseTxt> Ptxt::operator*(double other) {
  return std::make_shared<Ptxt>(*value_ * other,
                                name_ + " * " + std::to_string(other));
}

Ptxt& Ptxt::operator+=(double other) {
  value_->addInPlace(other);
  return *this;
}

Ptxt& Ptxt::operator*=(double other) {
  value_->multInPlace(other);
  return *this;
}

}  // namespace aluminum_shark
