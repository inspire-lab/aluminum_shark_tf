#include "tensorflow/compiler/plugin/aluminum_shark/utils/utils.h"

namespace aluminum_shark {

std::ostream& operator<<(std::ostream& os, const ShapePrint& shape) {
  os << "shape: " << shape.shape;
  return os;
}

// helpers
size_t multi_index_to_flat(const std::vector<size_t>& index,
                           const Shape& shape) {
  if (shape.size() != index.size()) {
    AS_LOG_S << "missmatching index: ";
    if (log()) {
      stream_vector(index);
    }
    AS_LOG_SA << " and shape: ";
    if (log()) {
      stream_vector(shape);
    }
    AS_LOG_SA << std::endl;
    throw std::invalid_argument("index and shape missmatch");
  }
  // const size_t = index.size();
  size_t ret = 0;
  size_t scale = 1;
  for (size_t dim = index.size(); dim-- != 0;) {
    if (index[dim] >= shape[dim]) {
      AS_LOG_S << "index: " << index << " and shape " << ShapePrint(shape)
               << " missmatch " << std::endl;
      throw std::invalid_argument("index and shape missmatch");
    }
    ret += scale * index[dim];
    scale *= shape[dim];
  }
  return ret;
}

void next_index(std::vector<size_t>& index, const Shape& shape) {
  // TODO RP
}

}  // namespace aluminum_shark