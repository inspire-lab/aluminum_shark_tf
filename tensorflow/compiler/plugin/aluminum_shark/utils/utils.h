#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_UTILS_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_UTILS_H

#include <iomanip>
#include <ios>
#include <iostream>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

namespace aluminum_shark {

template <class T>
std::ostream& stream_vector(const std::vector<T>& vec, size_t n = 0,
                            std::ostream& os = std::cout) {
  n = n == 0 || n > vec.size() ? vec.size() : n;
  os << "[";
  if (n != vec.size()) {
    for (size_t i = 0; i < n / 2; i++) {
      os << vec[i] << ", ";
    }
    os << "...";
    for (size_t i = vec.size() - n / 2; i < vec.size(); i++) {
      os << vec[i];
      if (i != vec.size() - 1) {
        os << ", ";
      }
    }
  } else {
    for (size_t i = 0; i < vec.size(); i++) {
      os << vec[i];
      if (i != vec.size() - 1) {
        os << ", ";
      }
    }
  }
  os << "]";
  return os;
}

template <class T>
void print_vector(const std::vector<T>& vec, size_t n = 0,
                  std::ostream& os = std::cout) {
  stream_vector<T>(vec, n, os) << std::endl;
}

using Shape = std::vector<size_t>;

// converts a python sytle index [x,y,z] into single index
// into the flat sotrage vector
size_t multi_index_to_flat(const std::vector<size_t>& index,
                           const Shape& shape);

// wrapper that allows us to overload << for shape only
class ShapePrint {
 public:
  ShapePrint(const Shape& s) : shape(s){};
  const Shape& shape;
};

std::ostream& operator<<(std::ostream& os, const ShapePrint& shape);

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    os << vec[i] << (i != vec.size() - 1 ? ", " : "");
  }
  os << "]";
  return os;
}

// simlpe wrapper that pairs a vector with a shape for display purposes
template <class T>
class PrintWithShape {
 public:
  PrintWithShape(const std::vector<T>& vec,
                 const std::vector<size_t>& shape_vector)
      : vector(vec), shape(shape_vector){};
  const std::vector<T>& vector;
  const std::vector<size_t>& shape;
};

void next_index(std::vector<size_t>& index, const Shape& shape);

template <class T>
std::ostream& operator<<(std::ostream& os, const PrintWithShape<T>& printer) {
  const std::vector<T>& v = printer.vector;
  const Shape& s = printer.shape;
  // std::streamsize old_size = os.prec
  // opening [
  // AS_LOG_S << "number of dims: " << s.size() << std::endl;
  for (size_t i = 0; i < s.size(); i++) {
    os << "[";
  }
  // so we can start a index 1
  if (v.size() > 0) {
    os << " " << std::setw(12) << v[0] << ", ";
  }
  // print the data
  for (size_t i = 1; i < printer.vector.size(); i++) {
    bool nl = false;
    size_t closing_brackets = 0;
    int div = 1;
    // check if we need to make closing brackets, calculating the product over
    // the dimensions. if it any point it divides the index without remainder we
    // need a closing ]. if we have any ] we also need a \n
    for (size_t j = s.size(); j-- != 0;) {
      div *= s[j];
      closing_brackets += (i % div == 0);  // true converts to 1, false to zero
    }
    // AS_LOG_S << "closing_brackets: " << closing_brackets << std::endl;
    for (size_t j = 0; j < closing_brackets; j++) {
      os << "]";
    }
    // closing brackets imply a new line
    if (closing_brackets != 0) {
      os << "\n";
    }
    // open up brackets again
    for (size_t j = 0; j < closing_brackets; j++) {
      os << "[";
    }
    // when we insert a brackets we need to adjust the the width so that it
    // lines up. s.size() is the max number of brackets. so we subtract the
    // number of brackets we inserted. in the end add 1 so things are not too
    // crowded
    size_t w = (closing_brackets != 0) * (s.size() - closing_brackets) + 1;
    os << std::setw(12 + w) << v[i] << ",";
  }

  for (auto i : s) {
    os << "]";
  }
  os << "\n";
  return os;
}

}  // namespace aluminum_shark
#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_UTILS_H \
        */
