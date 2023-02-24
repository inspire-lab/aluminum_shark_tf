#include "tensorflow/compiler/plugin/aluminum_shark/compiler.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/division_replacer.h"
#include "tensorflow/compiler/plugin/aluminum_shark/executable.h"
#include "tensorflow/compiler/plugin/aluminum_shark/hlo_evaluator.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
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
  pipeline.AddPass<BatchNormExpander>(
      /*rewrite_training_op=*/false,
      /*rewrite_inference_op=*/true,
      /*rewrite_grad_op=*/false);
  pipeline.AddPass<DivisionReplacer>();
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

  // find inplace ops
  evaluator->set_inplace_ops(FindInplaceOps(hlo_module.get()));

  evaluator->set_memory_dependencies(BuildMemoryDepencies(hlo_module.get()));

  // precomupte all ops with known inputs
  Precompute(hlo_module.get(), evaluator.get());

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

std::map<const HloInstruction*, std::unordered_set<const HloInstruction*>>
AluminumSharkCompiler::BuildMemoryDepencies(HloModule* module) {
  AS_LOG_INFO << "building memory dependency map" << std::endl;
  // a map that maps each instruction to a set of instuction that depend on it
  // at execution time an instruction is removed from the set once it is
  // evaluated. once the set is item the the key instruction can be freed
  std::map<const HloInstruction*, std::unordered_set<const HloInstruction*>>
      deps;

  // iterate over all nodes
  HloInstruction* root = module->entry_computation()->root_instruction();
  std::unordered_set<const HloInstruction*> nodes;
  nodes.insert(root);
  while (nodes.size() != 0) {
    // get first node from the set of unvisted nodes and remove
    auto node_iter = nodes.begin();
    const HloInstruction* node = *node_iter;
    AS_LOG_DEBUG << "visting " << node->name() << std::endl;
    nodes.erase(node_iter);

    // for each operand insert the current node into the
    auto n_operands = node->operand_count();
    for (size_t i = 0; i < n_operands; ++i) {
      const HloInstruction* operand = node->operand(i);
      // this should never happend but we don't want to add the root instruction
      // to the map. the root instruction ciphertext must not be deleted
      if (operand == root) {
        continue;
      }
      // add it to the iteration set
      nodes.insert(operand);
      // check if the operand is in the map
      auto iter = deps.find(operand);
      if (iter == deps.end()) {
        deps[operand] = {node};
      } else {
        iter->second.insert(node);
      }
    }
  }
  // log the memory map
  std::stringstream ss;
  for (auto iter : deps) {
    // get the set of operations and remove hlo
    const HloInstruction* key = iter.first;
    auto& op_set = iter.second;
    ss << "\t " << key->name() << " required for: ";
    for (auto op : op_set) {
      ss << op->name() << ", ";
    }
  }
  AS_LOG_DEBUG << "Memory depencies: " << ss.str() << std::endl;

  return deps;
}

std::unordered_set<const HloInstruction*> AluminumSharkCompiler::FindInplaceOps(
    HloModule* module) {
  AS_LOG_INFO << "finding inplace operations" << std::endl;
  // check for operations that can be done inplace
  HloInstruction* root = module->entry_computation()->root_instruction();

  // set of candidate ops
  std::unordered_set<const HloInstruction*> inplace_ops;
  // count the uses of HLOs
  std::map<const HloInstruction*, int> op_use_count;
  // iterate over all nodes
  std::unordered_set<const HloInstruction*> nodes;
  nodes.insert(root);
  std::unordered_set<const HloInstruction*> visited;
  while (nodes.size() != 0) {
    // get first node from the set of unvisted nodes and remove
    auto node_iter = nodes.begin();
    const HloInstruction* node = *node_iter;
    nodes.erase(node_iter);

    // collect a set of all operands
    auto n_operands = node->operand_count();
    std::unordered_set<const HloInstruction*> operand_set;
    for (size_t i = 0; i < n_operands; ++i) {
      operand_set.insert(node->operand(i));
    }

    // go over the unqiue operands and increase the use counter and add them to
    // the set to visit if did not visit them already
    for (auto n : operand_set) {
      // increase counter
      auto iter = op_use_count.find(n);
      if (iter == op_use_count.end()) {
        op_use_count[n] = 1;
      } else {
        iter->second += 1;
      }
      // check if we visited it already
      auto iter_v = visited.find(n);
      if (iter_v == visited.end()) {
        nodes.insert(n);
      }
    }

    // check if the node is a candidate op
    HloOpcode opcode = node->opcode();
    if (opcode == HloOpcode::kAdd          //
        || opcode == HloOpcode::kMultiply  //
        || opcode == HloOpcode::kSubtract  //
        || opcode == HloOpcode::kDivide    //
    ) {
      inplace_ops.insert(node);
    }

    // add current node to the list of visited notes
    visited.insert(node);
  }

  // logging
  if (::aluminum_shark::log(::aluminum_shark::AS_INFO)) {
    AS_LOG_INFO << "possible inplace ops: " << std::endl;
    for (const auto node : inplace_ops) {
      AS_LOG_SA << "\t" << node->name() << std::endl;
    }
    AS_LOG_INFO << "op use counts: " << std::endl;
    for (auto iter : op_use_count) {
      AS_LOG_SA << "\t" << iter.first->name() << ": " << iter.second
                << std::endl;
    }
  }

  // remove ops from the set are used more than once
  for (auto it = inplace_ops.begin(); it != inplace_ops.end();) {
    if (op_use_count[*it] != 1)
      it = inplace_ops.erase(it);
    else
      ++it;
  }
  if (::aluminum_shark::log(::aluminum_shark::AS_DEBUG)) {
    AS_LOG_DEBUG << "inplace ops:" << std::endl;
    for (const auto node : inplace_ops) {
      AS_LOG_SA << "\t" << node->name() << std::endl;
    }
  }
  return inplace_ops;
}

void AluminumSharkCompiler::Precompute(HloModule* module,
                                       AluminumSharkHloEvaluator* evaluator) {
  // generate print options
  HloPrintOptions print_options =
      HloPrintOptions()
          .set_print_subcomputation_mode(
              HloPrintOptions::PrintSubcomputationMode::kNonSequentialBodies)
          .set_print_metadata(false)
          // .set_print_backend_config(false)
          // .set_print_infeed_outfeed_config(false)
          // .set_print_only_essential_constants(true)
          .set_compact_operands(true)
          .set_print_operand_names(true)
          .set_print_operand_shape(true)
          // .set_print_operand_index_annotation_interval(0)
          // .set_print_program_shape(false)
          // .set_print_percent(false)
          // .set_print_control_dependencies(false)
          // .set_canonicalize_instruction_names(true)
          .set_print_ids(true)
          .set_indent_amount(2)
      // .set_canonicalize_computations(true)
      ;

  AS_LOG_DEBUG << "Running precomputation on: " << std::endl;
  AS_LOG_DEBUG << module->ToString(print_options) << std::endl;
  // get computation
  HloComputation* computation = module->entry_computation();

  // list of replacements
  std::vector<std::pair<HloInstruction*, std::unique_ptr<HloInstruction>>>
      replacements;

  do {
    // clear any leftovers
    replacements.clear();
    // get all instrunctions
    std::vector<HloInstruction*> instructions =
        computation->MakeInstructionPostOrder();

    // iterate over all instructions and evalute them if possible
    for (HloInstruction* hlo : instructions) {
      // if we have a constant or parameter. continue
      if (hlo->opcode() == HloOpcode::kParameter ||
          hlo->opcode() == HloOpcode::kConstant) {
        continue;
      }

      // check if all operands are constant
      for (HloInstruction* op : hlo->operands()) {
        if (op->opcode() != HloOpcode::kConstant) {
          // break out of this loop and contiue the outer for loop
          goto instruction_loop;
        }
      }

      // at this point we have an hlo that isn't a constant or parameter and has
      // only constant operands
      {
        // evaluate the hlo
        auto literal = evaluator->Evaluate(hlo).ConsumeValueOrDie();
        // create new constant hlo
        std::unique_ptr<HloInstruction> new_hlo =
            HloInstruction::CreateConstant(std::move(literal));
        // add it to the replacement list
        replacements.push_back(std::make_pair<>(hlo, std::move(new_hlo)));
      }

    instruction_loop:;  // to break the inner loop
    }
    // perform replacements
    for (auto& replacement : replacements) {
      computation->ReplaceWithNewInstruction(replacement.first,
                                             std::move(replacement.second));
    }

  } while (replacements.size() != 0);
  AS_LOG_DEBUG << "After precomputation: " << std::endl;
  AS_LOG_DEBUG << module->ToString(print_options) << std::endl;
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
