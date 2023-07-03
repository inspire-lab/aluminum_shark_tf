#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_ARG_UTILS_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_ARG_UTILS_H
#ifndef ALUMINUM_SHARK_COMMON_ARG_UTILS_H
#define ALUMINUM_SHARK_COMMON_ARG_UTILS_H

#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/he_backend/he_backend.h"

namespace aluminum_shark {

std::string args_to_string(const std::vector<aluminum_shark_Argument>& args);

std::string arg_to_string(const aluminum_shark_Argument& arg);

class illegal_arg_type : public std::exception {
 public:
  illegal_arg_type(const std::string& msg_ = "") : msg(msg_){};
  const std::string msg;
  const char* what() const noexcept override { return msg.c_str(); };
};

class arg_not_found : public std::exception {
 public:
  arg_not_found(const std::string& msg_ = "") : msg(msg_){};
  const std::string msg;
  const char* what() const noexcept override { return msg.c_str(); };
};

template <typename T>
uint get_type_id() {
  throw illegal_arg_type("incompatible type");
};

template <>
inline uint get_type_id<long>() {
  return 0;
};

template <>
inline uint get_type_id<double>() {
  return 1;
};

template <>
inline uint get_type_id<const char*>() {
  return 2;
};

template <typename T>
struct is_vector {
  static constexpr bool value = false;
};

template <template <typename...> class C, typename U>
struct is_vector<C<U>> {
  static constexpr bool value = std::is_same<C<U>, std::vector<U>>::value;
};

template <typename T>
std::enable_if_t<is_vector<T>::value, T> get_arg(
    const std::string& id, const std::vector<aluminum_shark_Argument>& args) {
  // decode array
  uint type_id = get_type_id<typename T::value_type>();
  for (auto& arg : args) {
    const char* name = arg.name;
    if (std::strcmp(name, id.c_str()) == 0) {
      if (arg.type != type_id || !arg.is_array) {
        throw illegal_arg_type();
      }
      typename T::value_type* arr =
          reinterpret_cast<typename T::value_type*>(arg.array_);
      T vec(arr, arr + arg.size_);
      return vec;
    }
  }
  throw arg_not_found();
};

template <typename T>
std::enable_if_t<!is_vector<T>::value, T> get_arg(
    const std::string& id, const std::vector<aluminum_shark_Argument>& args) {
  throw illegal_arg_type();
};

// template specialliztions for none-vector types
template <>
inline long get_arg<long>(const std::string& id,
                          const std::vector<aluminum_shark_Argument>& args) {
  uint type_id = get_type_id<long>();
  for (auto& arg : args) {
    const char* name = arg.name;
    if (std::strcmp(name, id.c_str()) == 0) {
      if (arg.type != type_id || arg.is_array) {
        throw illegal_arg_type();
      }
      return arg.int_;
    }
  }
  throw arg_not_found();
};

template <>
inline double get_arg<double>(
    const std::string& id, const std::vector<aluminum_shark_Argument>& args) {
  uint type_id = get_type_id<double>();
  for (auto& arg : args) {
    const char* name = arg.name;
    if (std::strcmp(name, id.c_str()) == 0) {
      if (arg.type != type_id || arg.is_array) {
        throw illegal_arg_type();
      }
      return arg.double_;
    }
  }
  throw arg_not_found();
};

template <>
inline const char* get_arg<const char*>(
    const std::string& id, const std::vector<aluminum_shark_Argument>& args) {
  uint type_id = get_type_id<const char*>();
  for (auto& arg : args) {
    const char* name = arg.name;
    if (std::strcmp(name, id.c_str()) == 0) {
      if (arg.type != type_id || arg.is_array) {
        throw illegal_arg_type();
      }
      return arg.string_;
    }
    throw arg_not_found();
  }
};

template <typename T>
T get_arg_safe(const std::string& id,
               const std::vector<aluminum_shark_Argument>& args) {
  try {
    return get_arg<T>(id, args);
  } catch (const illegal_arg_type& e) {
    return T();
  } catch (const arg_not_found& e) {
    return T();
  }
};

template <typename T>
T get_arg_default(const std::string& id,
                  const std::vector<aluminum_shark_Argument>& args,
                  T default_value) {
  try {
    return get_arg<T>(id, args);
  } catch (const illegal_arg_type& e) {
    return default_value;
  } catch (const arg_not_found& e) {
    return default_value;
  }
};

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_COMMON_ARG_UTILS_H */

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PYTHON_ARG_UTILS_H \
        */
