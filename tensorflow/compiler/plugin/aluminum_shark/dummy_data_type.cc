#include "tensorflow/compiler/plugin/aluminum_shark/dummy_data_type.h"

#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

namespace aluminum_shark {

std::string DummyDataType::to_string() const {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < value_.size(); ++i) {
    ss << value_[i];
    if (i != value_.size() - 1) {
      ss << ",";
    }
  }
  ss << "] " << name_;
  return ss.str();
}

DummyDataType operator+(const DummyDataType& one, const DummyDataType& two) {
  DummyDataType ret;
  if (one.value_.size() != two.value_.size()) {
    const DummyDataType& scalar =
        one.value_.size() > two.value_.size() ? two : one;
    const DummyDataType& vector =
        one.value_.size() < two.value_.size() ? two : one;
    if (scalar.value_.size() != 1) {
      AS_LOG("invalid shapes for addition: " + one.to_string() + " and " +
             two.to_string());
      ret.name_ = "error during adding: " + one.name_ + " and " + two.name_;
    } else {
      ret = vector;
      for (size_t i = 0; i < ret.value_.size(); ++i) {
        ret.value_[i] += scalar.value_[0];
      }
      ret.name_ += " + " + scalar.name_;
    }
  } else {
    ret = one;
    for (size_t i = 0; i < ret.value_.size(); ++i) {
      ret.value_[i] += two.value_[i];
    }
    ret.name_ += " + " + two.name_;
  }
  return std::move(ret);
}

DummyDataType operator*(const DummyDataType& one, const DummyDataType& two) {
  DummyDataType ret;
  if (one.value_.size() != two.value_.size()) {
    const DummyDataType& scalar =
        one.value_.size() > two.value_.size() ? two : one;
    const DummyDataType& vector =
        one.value_.size() < two.value_.size() ? two : one;
    if (scalar.value_.size() != 1) {
      AS_LOG("invalid shapes for multiplication: " + one.to_string() + " and " +
             two.to_string());
      ret.name_ =
          "error during multiplication: " + one.name_ + " and " + two.name_;
    } else {
      ret = vector;
      for (size_t i = 0; i < ret.value_.size(); ++i) {
        ret.value_[i] += scalar.value_[0];
      }
      ret.name_ += " * " + scalar.name_;
    }
  } else {
    ret = one;
    for (size_t i = 0; i < ret.value_.size(); ++i) {
      ret.value_[i] += two.value_[i];
    }
    ret.name_ += " * " + two.name_;
  }
  return std::move(ret);
}

void DummyDataType::setValue(std::vector<long>&& value) {
  value_ = std::move(value);
};

void DummyDataType::setName(const std::string& name) { name_ = name; }

const std::vector<long>& DummyDataType::getValue() const { return value_; }
const std::string& DummyDataType::getName() const { return name_; }

}  // namespace aluminum_shark
