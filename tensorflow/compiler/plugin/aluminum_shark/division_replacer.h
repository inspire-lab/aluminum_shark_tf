#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DIVISION_REPLACER_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DIVISION_REPLACER_H

#include "tensorflow/compiler/xla/service/op_expander_pass.h"

namespace xla {
namespace aluminum_shark {

// Replaces division by multiplication with the inverse
class DivisionReplacer : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "division_replacer"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace aluminum_shark
}  // namespace xla

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_DIVISION_REPLACER_H \
        */
