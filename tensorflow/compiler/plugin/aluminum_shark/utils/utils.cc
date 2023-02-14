#include "tensorflow/compiler/plugin/aluminum_shark/utils/utils.h"

namespace aluminum_shark {

std::ostream& operator<<(std::ostream& os, const ShapePrint& shape) {
  os << "shape: " << shape.shape;
  return os;
}

void next_index(std::vector<size_t>& index, const Shape& shape) {
  // TODO RP
}

}  // namespace aluminum_shark