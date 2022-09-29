#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_COMPILER_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_COMPILER_H

#include <memory>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/hlo_evaluator.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/plugin/aluminum_shark/platform_id.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace xla {
namespace aluminum_shark {

// This is an old comment but it might helpful:
// Despite the inherited "compiler" naming, InterpreterCompiler does not
// perform any lowering as other backends do. It operates at HLO-level for
// and is responsible for generating an InterpreterExecutable.
// Refer to interpreter/README.md for more.
class AluminumSharkCompiler : public Compiler {
 public:
  AluminumSharkCompiler() { AS_LOG("instantiating aluminum shark compiler"); }
  ~AluminumSharkCompiler() override {}

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;
  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;
  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& aot_options) override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  se::Platform::Id PlatformId() const override;

  std::unordered_set<const HloInstruction*> FindInplaceOps(HloModule* module);

  void Precompute(HloModule* module, AluminumSharkHloEvaluator* evaluator);

 private:
  Status RunHloOptimization(HloModule* hlo_module);

  TF_DISALLOW_COPY_AND_ASSIGN(AluminumSharkCompiler);
};

}  // namespace aluminum_shark
}  // namespace xla

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_COMPILER_H \
        */
