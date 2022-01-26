#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"

#include <cxxabi.h>

#include <typeinfo>

#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

namespace aluminum_shark {

std::string Ctxt::to_string() const {
  if (!value_.get()) {
    return "Ciphertext is not initialized.";
  }
  // TODO: get more info
  return name_ + value_->to_string();
}

// Ctxt operator+(const Ctxt& one, const Ctxt& two) {
//   // TODO: shape checking?
//   Ctxt ret(one + two, one.getName() + " + " + two.getName());
//   return std::move(ret);
// }

// Ctxt operator*(const Ctxt& one, const Ctxt& two) {
//   Ctxt ret(one * two, one.getName() + " * " + two.getName());
//   return std::move(ret);
// }

Ctxt::Ctxt(std::shared_ptr<HECtxt> hectxt, std::string name)
    : value_(hectxt), name_(name) {
  AS_LOG_S << "Created Ctxt " << name << " internal "
           << reinterpret_cast<void*>(value_.get()) << std::endl;
}

// create a deep copy which also creates a copy of stored object
Ctxt Ctxt::deepCopy() const {
  AS_LOG("creating deep copy of: " + name_);
  Ctxt copy = *this;
  // create a copy of the stored object
  copy.setValue(std::shared_ptr<HECtxt>(value_->deepCopy()));
  return copy;
}

void Ctxt::setName(const std::string& name) { name_ = name; }
const std::string& Ctxt::getName() const { return name_; }
void Ctxt::setValue(std::shared_ptr<HECtxt> value_ptr) {
  AS_LOG_S << "set HECtxt " << reinterpret_cast<void*>(value_ptr.get())
           << std::endl;
  value_ = value_ptr;
}

const HECtxt& Ctxt::getValue() const { return *value_; }
HECtxt& Ctxt::getValue() { return *value_; }
std::shared_ptr<HECtxt> Ctxt::getValuePtr() const { return value_; }

// // arithmetic opertions
// // Ctxt x Ctxt operations
// Ctxt operator+(const Ctxt& one, const Ctxt& two) {
//   Ctxt ret = one.deepCopy();
//   ret += two;
//   ret.setName(ret.getName() + " + " + two.getName());
//   return two;
// }
// Ctxt operator*(const Ctxt& one, const Ctxt& two) {
//   Ctxt ret = one.deepCopy();
//   ret *= two;
//   ret.setName(ret.getName() + " * " + two.getName());
//   return two;
// }
// Ctxt& operator+=(const Ctxt& two) {
//   // TODO: update name or not?
//   value_->addInPlace(two.value_);
//   return *this;
// }
// Ctxt& operator*=(const Ctxt& two) {
//   // TODO: update name or not?
//   value_->multInPlace(two.value_);
//   return *this;
// }

// // Ctxt x Ptxt operations
// Ctxt operator+(const Ctxt& one, const Ptxt& two) {
//   Ctxt ret = one.deepCopy();
//   ret += two;
//   ret.setName(ret.getName() + " + " + two.getName());
//   return two;
// }
// Ctxt operator*(const Ctxt& one, const Ptxt& two) {
//   Ctxt ret = one.deepCopy();
//   ret *= two;
//   ret.setName(ret.getName() + " * " + two.getName());
//   return two;
// }
// Ctxt& operator+=(const Ptxt& two) {
//   // TODO: update name or not?
//   value_->addInPlace(two.value_);
//   return *this;
// }
// Ctxt& operator*=(const Ptxt& two) {
//   // TODO: update name or not?
//   value_->multInPlace(two.value_);
//   return *this;
// }

// // Ctxt x integral operations
// // integer
// Ctxt operator+(const Ctxt& one, long two) {
//   Ctxt ret = one.deepCopy();
//   ret += two;
//   ret.setName(ret.getName() + " + " + std::string(two));
//   return two;
// }
// Ctxt operator*(const Ctxt& one, long two) {
//   Ctxt ret = one.deepCopy();
//   ret *= two;
//   ret.setName(ret.getName() + " * " + std::string(two));
//   return two;
// }
// Ctxt& operator+=(long two) {
//   // TODO: update name or not?
//   value_->addInPlace(two);
//   return *this;
// }
// Ctxt& operator*=(long two) {
//   // TODO: update name or not?
//   value_->multInPlace(two);
//   return *this;
// }
// // floating point
// Ctxt operator+(const Ctxt& one, double two) {
//   Ctxt ret = one.deepCopy();
//   ret += two;
//   ret.setName(ret.getName() + " + " + std::string(two));
//   return two;
// }
// Ctxt operator*(const Ctxt& one, double two) {
//   Ctxt ret = one.deepCopy();
//   ret *= two;
//   ret.setName(ret.getName() + " * " + std::string(two));
//   return two;
// }

// BaseTxttxt interface
std::shared_ptr<BaseTxt> Ctxt::operator+(const BaseTxt& other) const {
  int code = -4;
  char* buffer = abi::__cxa_demangle(typeid(other).name(), NULL, NULL, &code);
  AS_LOG_S << "typeinfo: " << buffer << std::endl;
  try {
    const Ctxt& ctxt = dynamic_cast<const Ctxt&>(other);
    AS_LOG_S << "adding other ctxt " << ctxt.getName() << std::endl;
    HECtxt& this_value = *value_;
    AS_LOG_S << "Gotten this value" << std::endl;
    HECtxt* other_value = ctxt.getValuePtr().get();
    AS_LOG_S << "Gotten other value" << std::endl;
    HECtxt* he_ctxt = this_value + other_value;
    AS_LOG_S << "did some multiplication" << std::endl;
    auto ptr = std::make_shared<Ctxt>(*value_ + ctxt.getValuePtr().get(),
                                      name_ + " + " + ctxt.getName());
    AS_LOG_S << "created new ciphtertext " << ptr->getName() << std::endl;
    return ptr;
  } catch (const std::bad_cast& e) {
    /* this is ok. ignore */
  }
  try {
    const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
    AS_LOG_S << "adding other ptxt " << ptxt.getName() << std::endl;
    return std::make_shared<Ctxt>(*value_ + ptxt.getValuePtr().get(),
                                  name_ + " + " + ptxt.getName());
  } catch (const std::bad_cast& e) {
    // this is not ok
    AS_LOG_S << e.what() << std::endl;
    throw e;
  }
}
std::shared_ptr<BaseTxt> Ctxt::operator*(const BaseTxt& other) const {
  int code = -4;
  char* buffer = abi::__cxa_demangle(typeid(other).name(), NULL, NULL, &code);
  AS_LOG_S << "typeinfo: " << buffer << std::endl;
  free(buffer);
  try {
    const Ctxt& ctxt = dynamic_cast<const Ctxt&>(other);
    // AS_LOG_S << "multiplying with other ctxt" << ctxt.getName() << std::endl;
    // HECtxt& this_value = *value_;
    // AS_LOG_S << "Gotten this value" << std::endl;
    // HECtxt* other_value = ctxt.getValuePtr().get();
    // AS_LOG_S << "Gotten other value" << std::endl;
    // HECtxt* he_ctxt = this_value * other_value;
    // AS_LOG_S << "did some multiplication" << std::endl;
    auto ptr = std::make_shared<Ctxt>(*value_ * ctxt.getValuePtr().get(),
                                      name_ + " * " + ctxt.getName());
    AS_LOG_S << "created new ciphtertext " << ptr->getName() << std::endl;
    return ptr;
  } catch (const std::bad_cast& e) {
    /* this is ok. can be ignore */
    AS_LOG_S << e.what() << std::endl;
  } catch (const std::exception& e) {
    /* this is ok. ignore */
    AS_LOG_S << "Some other exception occurd: " << e.what() << std::endl;
  }
  try {
    const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
    AS_LOG_S << "multiplying with other ptxt" << ptxt.getName() << std::endl;
    return std::make_shared<Ctxt>(*value_ * ptxt.getValuePtr().get(),
                                  name_ + " * " + ptxt.getName());
  } catch (const std::bad_cast& e) {
    // this is not ok
    AS_LOG_S << e.what() << std::endl;
    throw e;
  }
}

Ctxt& Ctxt::operator+=(const BaseTxt& other) {
  try {
    const Ctxt& ctxt = dynamic_cast<const Ctxt&>(other);
    value_->addInPlace(ctxt.getValuePtr().get());
    return *this;
  } catch (const std::bad_cast& e) {
    /* this is ok. ignore */
  }
  try {
    const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
    value_->addInPlace(ptxt.getValuePtr().get());
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
    value_->multInPlace(ctxt.getValuePtr().get());
    return *this;
  } catch (const std::bad_cast& e) {
    /* this is ok. ignore */
  }
  const Ptxt& ptxt = dynamic_cast<const Ptxt&>(other);
  value_->multInPlace(ptxt.getValuePtr().get());
  return *this;
}

// integer
std::shared_ptr<BaseTxt> Ctxt::operator+(long other) {
  return std::make_shared<Ctxt>(std::shared_ptr<HECtxt>(*value_ + other),
                                name_ + " + " + std::to_string(other));
}

std::shared_ptr<BaseTxt> Ctxt::operator*(long other) {
  return std::make_shared<Ctxt>(std::shared_ptr<HECtxt>(*value_ * other),
                                name_ + " * " + std::to_string(other));
}

Ctxt& Ctxt::operator+=(long other) {
  value_->addInPlace(other);
  return *this;
}

Ctxt& Ctxt::operator*=(long other) {
  value_->multInPlace(other);
  return *this;
}

// floating point
std::shared_ptr<BaseTxt> Ctxt::operator+(double other) {
  return std::make_shared<Ctxt>(std::shared_ptr<HECtxt>(*value_ + other),
                                name_ + " + " + std::to_string(other));
}

std::shared_ptr<BaseTxt> Ctxt::operator*(double other) {
  return std::make_shared<Ctxt>(std::shared_ptr<HECtxt>(*value_ * other),
                                name_ + " * " + std::to_string(other));
}

Ctxt& Ctxt::operator+=(double other) {
  value_->addInPlace(other);
  return *this;
}

Ctxt& Ctxt::operator*=(double other) {
  // TODO: update name or not?
  value_->multInPlace(other);
  return *this;
}

}  // namespace aluminum_shark
