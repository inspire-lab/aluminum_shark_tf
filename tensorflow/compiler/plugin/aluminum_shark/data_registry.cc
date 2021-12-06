#include "tensorflow/compiler/plugin/aluminum_shark/data_registry.h"

namespace aluminum_shark {

DummyDataType& DataRegistry::get(xla::HLOInstruction* instruction) {
  return map_[instruction];
}

void DataRegistry::put(xla::HLOInstruction* instruction, DummyDataType&& ddt) {
  return map_[instruction] = ddt;
}

bool exists(const xla::HLOInstruction* instruction) {
  return map_.contains(instruction);
}

}  // namespace aluminum_shark
