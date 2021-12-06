#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_ALUMINUM_SHARK_DEF_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_ALUMINUM_SHARK_DEF_H

#include <array>

#include "tensorflow/core/framework/types.pb.h"

namespace aluminum_shark {

extern const char* const DEVICE_XLA_HE;
extern const char* const DEVICE_XLA_HE_JIT;

constexpr std::array<tensorflow::DataType, 14> kAllXlaHETypes = {
    {tensorflow::DT_UINT8, tensorflow::DT_QUINT8, tensorflow::DT_UINT16,
     tensorflow::DT_INT8, tensorflow::DT_QINT8, tensorflow::DT_INT16,
     tensorflow::DT_INT32, tensorflow::DT_QINT32, tensorflow::DT_INT64,
     tensorflow::DT_HALF, tensorflow::DT_FLOAT, tensorflow::DT_DOUBLE,
     tensorflow::DT_BOOL, tensorflow::DT_BFLOAT16}};

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_ALUMINUM_SHARK_DEF_H \
        */
