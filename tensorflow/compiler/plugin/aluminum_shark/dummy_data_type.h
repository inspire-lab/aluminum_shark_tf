#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DUMMY_DATA_TYPE_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DUMMY_DATA_TYPE_H

#include <sstream>
#include <string>
#include <vector>

namespace aluminum_shark {

class DummyDataType {
 public:
  DummyDataType() {}
  DummyDataType(std::string name) : name_(name) {}
  DummyDataType(std::vector<long> v, std::string name)
      : value_(v), name_(name) {}
  //   DummyDataType(DummyDataType&& other)
  //       : value_(std::move(other.value_)), name_(std::move(other.name_)) {}

  std::string to_string() const;

  friend DummyDataType operator+(const DummyDataType& one,
                                 const DummyDataType& two);
  friend DummyDataType operator*(const DummyDataType& one,
                                 const DummyDataType& two);

  void setValue(std::vector<long>&& value);
  void setName(const std::string& name);

  const std::vector<long>& getValue() const;
  const std::string& getName() const;

  //   const std::string deugbInfo() { return " "; }

 private:
  std::vector<long> value_;
  std::string name_;
};

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DUMMY_DATA_TYPE_H \
        */
