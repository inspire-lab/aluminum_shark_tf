#include "arg_utils.h"

#include <sstream>

namespace aluminum_shark {

void arg_to_stream(const aluminum_shark_Argument& arg,
                   std::stringstream& stream) {
  // add name
  stream << arg.name;

  // add type
  stream << "\n  type: ";
  if (arg.type == 0) {
    stream << "0; int";
  } else if (arg.type == 1) {
    stream << "1; double";
  } else if (arg.type == 2) {
    stream << "2; string";
  } else {
    stream << arg.type << "; unkown";
  }
  stream << "\n  array: " << arg.array_;
  if (arg.array_) {
    stream << " size: " << arg.size_;
  }
  stream << "\n";
  // add single value
  if (!arg.array_) {
    stream << "  value: ";
    switch (arg.type) {
      case 0:
        stream << arg.int_;
        break;
      case 1:
        stream << arg.double_;
        break;
      case 2:
        stream << arg.string_;
        break;
      default:
        stream << "?";
        break;
    }
    stream << "\n";
  } else {  // add array
    stream << "  values: [\n";
    if (arg.type == 0) {
      long* arr = reinterpret_cast<long*>(arg.array_);
      for (size_t i = 0; i < arg.size_; ++i) {
        stream << arr[i];
        if (i != arg.size_ - 1) {
          stream << ",";
        }
        stream << "\n";
      }
      stream << "          ]\n";
    } else if (arg.type == 1) {
      double* arr = reinterpret_cast<double*>(arg.array_);
      for (size_t i = 0; i < arg.size_; ++i) {
        stream << arr[i];
        if (i != arg.size_ - 1) {
          stream << ",";
        }
        stream << "\n";
      }
      stream << "          ]\n";
    } else if (arg.type == 2) {
      const char* arr = reinterpret_cast<const char*>(arg.array_);
      for (size_t i = 0; i < arg.size_; ++i) {
        stream << arr[i];
        if (i != arg.size_ - 1) {
          stream << ",";
        }
        stream << "\n";
      }
      stream << "          ]\n";
    } else {
      stream << "?]\n";
    }
  }
}

std::string args_to_string(const std::vector<aluminum_shark_Argument>& args) {
  std::stringstream stream;
  for (const auto& a : args) {
    arg_to_stream(a, stream);
  }
  return stream.str();
}

std::string arg_to_string(const aluminum_shark_Argument& arg) {
  std::stringstream stream;
  arg_to_stream(arg, stream);
  return stream.str();
}

}  // namespace aluminum_shark