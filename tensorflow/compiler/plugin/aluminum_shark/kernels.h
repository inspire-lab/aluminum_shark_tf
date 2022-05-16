#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_KERNELS_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_KERNELS_H

#include <memory>

#include "tensorflow/compiler/plugin/aluminum_shark/base_txt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/layout.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/plugin/aluminum_shark/utils/utils.h"

namespace aluminum_shark {

namespace kernel_helpers {

bool matrix_mult_shape_compatible(const Shape& lhs, const Shape& rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  if (lhs_size != rhs_size) {
    return false;
  }
  if (lhs_size < 2) {
    return false;
  }
  size_t s = lhs_size;
  return lhs[s - 1] == rhs[s - 2];
};

}  // namespace kernel_helpers

namespace simple {

void matrix_multiplication(BaseTxt& lhs, BaseTxt& rhs,
                           std::shared_ptr<BaseTxt> result) {
  if (!kernel_helpers::matrix_mult_shape_compatible(lhs.shape(), rhs.shape())) {
    AS_LOG_S << "Incompatible matrix shapes " << stream_vector(lhs.shape())
             << " and " stream_vector(lhs.shape()) << std::endl;
    throw std::runtime_error("matrix shape mismatch");
  }
}

}  // namespace simple

namespace batch {

void matrix_multiplication(BaseTxt& lhs, BaseTxt& rhs,
                           std::shared_ptr<BaseTxt> result);

}

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_KERNELS_H \
        */
