#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DATA_REGISTRY_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DATA_REGISTRY_H

#include <map>

#include "tensorflow/compiler/plugin/aluminum_shark/dummy_data_type.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace aluminum_shark {

class DataRegistry {
 public:
  static DataRegistry& getInstance() {
    AS_LOG("getting DataRegistry instance");
    static DataRegistry instance;
    return instance;
  }

  DummyDataType& get(xla::HLOInstruction* instruction);
  void put(xla::HLOInstruction* instruction, DummyDataType&& ddt);
  bool exists(xla::HLOInstruction* instruction);

 private:
  std::map<xla::HLOInstruction*, DummyDataType> map_;
}

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DATA_REGISTRY_H \
        */
