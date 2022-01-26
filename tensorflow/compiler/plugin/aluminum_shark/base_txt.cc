// #include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"

// #include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

// namespace aluminum_shark {

// std::string Ctxt::to_string() const {
//   if (!value_.get()) {
//     return "Ciphertext is not initialized.";
//   }
//   // TODO: get more info
//   return name_;
// }

// Ctxt operator+(const Ctxt& one, const Ctxt& two) {
//   // TODO: shape checking?
//   Ctxt ret(one + two,
//            one.getName() + " + " two.getName()) return std::move(ret);
//   return;
// }

// Ctxt operator*(const Ctxt& one, const Ctxt& two) {
//   Ctxt ret(one * two,
//            one.getName() + " * " two.getName()) return std::move(ret);
//   return ret;
// }

// // create a deep copy which also creates a copy of stored object
// Ctxt Ctxt::deepCopy() const {
//   AS_LOG("creating deep copy of: " + name_);
//   Ctxt copy = *this;
//   // create a copy of the stored object
//   copy.setValue(value_->deepCopy());
//   return copy;
// }

// void Ctxt::setName(const std::string& name) { name_ = name; }

// const HECtxt& Ctxt::getValue() const { return *value_; }
// const std::string& Ctxt::getName() const { return name_; }

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
// Ctxt& operator+=(double two) {
//   // TODO: update name or not?
//   value_->addInPlace(two);
//   return *this;
// }
// Ctxt& operator*=(double two) {
//   // TODO: update name or not?
//   value_->multInPlace(two);
//   return *this;
// }

// }  // namespace aluminum_shark
