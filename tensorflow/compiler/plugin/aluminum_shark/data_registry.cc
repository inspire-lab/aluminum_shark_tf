#include "tensorflow/compiler/plugin/aluminum_shark/data_registry.h"

namespace aluminum_shark {

Ctxt& DataRegistry::get(xla::HLOInstruction* instruction) {
  return map_[instruction];
}

void DataRegistry::put(xla::HLOInstruction* instruction, Ctxt&& ctxt) {
  return map_[instruction] = ctxt;
}

bool exists(const xla::HLOInstruction* instruction) {
  return map_.contains(instruction);
}

}  // namespace aluminum_shark
