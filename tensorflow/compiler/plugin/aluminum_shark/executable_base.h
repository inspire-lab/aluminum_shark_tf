#ifndef TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_EXECUTABLE_BASE_H
#define TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_EXECUTABLE_BASE_H

#include <memory>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/dynamic_dimension_inference.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla.pb.h"
namespace xla {
namespace aluminum_shark {

// Responsible for running a HLO graph through the HloEvaluator and output
// buffer allocation. Refer to interpreter/README.md for more.
class AluminumSharkExecutableBase : public Executable {
 public:
  explicit AluminumSharkExecutableBase(std::unique_ptr<HloModule> hlo_module);

  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

 protected:
  virtual StatusOr<Literal> Evaluate(
      const ServiceExecutableRunOptions* run_options,
      const HloComputation& computation,
      absl::Span<const Literal> arg_literals) = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(AluminumSharkExecutableBase);
};

}  // namespace aluminum_shark
}  // namespace xla

#endif /*TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_EXECUTABLE_BASE_H */
