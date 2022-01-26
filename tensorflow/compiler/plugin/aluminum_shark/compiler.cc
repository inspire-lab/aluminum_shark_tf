#include "tensorflow/compiler/plugin/aluminum_shark/compiler.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/executable.h"
#include "tensorflow/compiler/plugin/aluminum_shark/hlo_evaluator.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/comparison_expander.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/eigh_expander.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/qr_expander.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace aluminum_shark {

namespace {

// Handles custom_call ops during evaluation by routing them through the global
// CPU registry used by other CPU-based backends.
StatusOr<Literal> HandleEvaluatorCustomCall(
    HloInstruction* custom_call, absl::Span<const Literal*> operands) {
  // Find the target C function in the global registry.
  auto* registry = CustomCallTargetRegistry::Global();
  void* target_fn = registry->Lookup(custom_call->custom_call_target(), "Host");
  if (!target_fn) {
    return NotFound("Custom call target '%s' was not registered",
                    custom_call->custom_call_target());
  }

  // Populate pointers to operand and output literal data.
  std::vector<const void*> operand_data;
  operand_data.reserve(operands.size());
  for (const auto* literal : operands) {
    operand_data.push_back(literal->untyped_data());
  }
  auto output = Literal::CreateFromShape(custom_call->shape());
  void* output_data = output.untyped_data();

  // Call the target function matching the C ABI used by the CPU backends.
  auto* typed_fn = reinterpret_cast<void (*)(void*, const void**)>(target_fn);
  (*typed_fn)(output_data, operand_data.data());

  return std::move(output);
}

}  // namespace

Status AluminumSharkCompiler::RunHloOptimization(HloModule* hlo_module) {
  AS_LOG("Running hlo optimizations " + hlo_module->name());
  HloPassPipeline pipeline("Aluminum Shark");

  pipeline.AddPass<DynamicIndexSplitter>();
  pipeline.AddPass<CholeskyExpander>();
  pipeline.AddPass<QrExpander>();
  pipeline.AddPass<EighExpander>();
  pipeline.AddPass<ComparisonExpander>();
  pipeline.AddPass<TriangularSolveExpander>();
  pipeline.AddPass<LayoutAssignment>(
      hlo_module->mutable_entry_computation_layout());

  return pipeline.Run(hlo_module).status();
}

StatusOr<std::unique_ptr<HloModule>> AluminumSharkCompiler::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* /*stream_exec*/,
    const CompileOptions& /*options*/) {
  AS_LOG("Run hlo passes on graph " + hlo_module->name());
  TF_RETURN_IF_ERROR(RunHloOptimization(hlo_module.get()));
  return std::move(hlo_module);
}

StatusOr<std::unique_ptr<Executable>> AluminumSharkCompiler::RunBackend(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& /*options*/) {
  TF_RET_CHECK(stream_exec != nullptr);

  AS_LOG("Run aluminum shark backend " + hlo_module->name());

  TF_ASSIGN_OR_RETURN(DynamicDimensionInference dynamic_dimension_inference,
                      DynamicDimensionInference::Run(hlo_module.get()));

  auto evaluator = absl::make_unique<AluminumSharkHloEvaluator>();
  evaluator->set_use_fast_path(
      hlo_module->config().debug_options().xla_hlo_evaluator_use_fast_path());
  evaluator->set_custom_call_handler(HandleEvaluatorCustomCall);

  // Create executable from only the Hlo module.
  std::unique_ptr<Executable> executable =
      absl::make_unique<AluminumSharkExecutable>(
          std::move(hlo_module), std::move(evaluator),
          std::move(dynamic_dimension_inference));

  return std::move(executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>>
AluminumSharkCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_exec,
    const CompileOptions& options) {
  AS_LOG("Running aluminum shark compiler");
  if (module_group->empty()) {
    return std::vector<std::unique_ptr<Executable>>();
  }
  if (module_group->size() > 1) {
    return tensorflow::errors::Unimplemented(
        "Compilation of multiple HLO modules is not supported on "
        "AluminumShark.");
  }
  if (stream_exec.size() != 1 || stream_exec[0].size() != 1) {
    return tensorflow::errors::Unimplemented(
        "Unexpected numbererss of StreamExecutor's.");
  }
  auto hlo_modules = module_group->ConsumeModules();
  TF_ASSIGN_OR_RETURN(auto module, RunHloPasses(std::move(hlo_modules[0]),
                                                stream_exec[0][0], options));
  TF_ASSIGN_OR_RETURN(auto executable, RunBackend(std::move(module),
                                                  stream_exec[0][0], options));

  std::vector<std::unique_ptr<Executable>> ret;
  ret.push_back(std::move(executable));
  return std::move(ret);
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
AluminumSharkCompiler::CompileAheadOfTime(
    std::unique_ptr<HloModuleGroup> module_group,
    const AotCompilationOptions& aot_options) {
  return tensorflow::errors::InvalidArgument(
      "AOT compilation not supported on AluminumShark");
}

se::Platform::Id AluminumSharkCompiler::PlatformId() const {
  return se::aluminum_shark::kXlaAluminumSharkPlatformId;
}

HloCostAnalysis::ShapeSizeFunction
AluminumSharkCompiler::ShapeSizeBytesFunction() const {
  return AluminumSharkExecutable::ShapeSizeBytes;
}

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      se::aluminum_shark::kXlaAluminumSharkPlatformId, []() {
        return absl::make_unique<xla::aluminum_shark::AluminumSharkCompiler>();
      });
  xla::ComputationPlacer::RegisterComputationPlacer(
      se::aluminum_shark::kXlaAluminumSharkPlatformId,
      []() { return absl::make_unique<xla::ComputationPlacer>(); });
  return true;
}

static bool module_initialized = InitModule();

}  // namespace aluminum_shark
}  // namespace xla
