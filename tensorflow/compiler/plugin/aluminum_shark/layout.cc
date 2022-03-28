#include "tensorflow/compiler/plugin/aluminum_shark/layout.h"

#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

namespace aluminum_shark {

void SimpleLayout::init() {
  for (size_t i = 0; i < indicies_.size(); ++i) {
    indicies_[i] = std::vector<size_t>{i, 0};
  }
};

void BatchLayout::init() {
  size_t bs = shape_[0];  // assumes batch dim is first
  size_t step_size = size_ / bs;
  for (size_t i = 0; i < size_; ++i) {
    // put every batch dimension into a single ciphertext
    indicies_[i] = std::vector<size_t>{i % step_size, i / step_size};
  }
};

}  // namespace aluminum_shark