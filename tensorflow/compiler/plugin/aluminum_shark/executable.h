#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_EXECUTABLE_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_EXECUTABLE_H

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/compiler/plugin/aluminum_shark/executable_base.h"
#include "tensorflow/compiler/plugin/aluminum_shark/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace aluminum_shark {

// Responsible for running a HLO graph through the HloEvaluator and output
// buffer allocation. Refer to interpreter/README.md for more.
class AluminumSharkExecutable : public AluminumSharkExecutableBase {
 public:
  AluminumSharkExecutable(
      std::unique_ptr<HloModule> hlo_module,
      std::unique_ptr<AluminumSharkHloEvaluator> evaluator,
      absl::optional<DynamicDimensionInference> dynamic_dymension_inference);

  static int64_t ShapeSizeBytes(const Shape& shape);

 protected:
  StatusOr<Literal> Evaluate(const ServiceExecutableRunOptions* run_options,
                             const HloComputation& computation,
                             absl::Span<const Literal> arg_literals) override
      TF_LOCKS_EXCLUDED(evaluator_lock_);

  // The interpreter interprets executables with an HloEvaluator.
  std::unique_ptr<AluminumSharkHloEvaluator> evaluator_
      TF_PT_GUARDED_BY(evaluator_lock_);
  mutable tensorflow::mutex evaluator_lock_;

 private:
  absl::optional<DynamicDimensionInference> dynamic_dimension_inference_;
  TF_DISALLOW_COPY_AND_ASSIGN(AluminumSharkExecutable);
};

}  // namespace aluminum_shark
}  // namespace xla

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_EXECUTABLE_H \
        */
