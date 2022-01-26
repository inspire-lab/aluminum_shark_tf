#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_UTILS_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_UTILS_H
#include <iostream>
#include <vector>

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

}  // namespace aluminum_shark
#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_UTILS_UTILS_H \
        */
