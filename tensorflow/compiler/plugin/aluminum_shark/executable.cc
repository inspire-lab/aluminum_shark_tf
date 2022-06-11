#include "tensorflow/compiler/plugin/aluminum_shark/executable.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/executable_base.h"
#include "tensorflow/compiler/plugin/aluminum_shark/executor.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace aluminum_shark {

AluminumSharkExecutable::AluminumSharkExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<AluminumSharkHloEvaluator> evaluator,
    absl::optional<DynamicDimensionInference> dynamic_dymension_inference)
    : AluminumSharkExecutableBase(std::move(hlo_module)),
      evaluator_(std::move(evaluator)),
      dynamic_dimension_inference_(std::move(dynamic_dymension_inference)) {
  if (dynamic_dimension_inference_.has_value()) {
    evaluator_->set_dynamic_dimension_inference(
        &dynamic_dimension_inference_.value());
  }
}

StatusOr<Literal> AluminumSharkExecutable::Evaluate(
    const ServiceExecutableRunOptions* run_options,
    const HloComputation& computation, absl::Span<const Literal> arg_literals) {
  // Execute the graph using the HloEvaluator.
  tensorflow::mutex_lock lock(evaluator_lock_);

  // let's see what is in the computation
  std::stringstream ss;

  AS_LOG("HloComputation: " + computation.ToString());

  evaluator_->ResetVisitStates();

  if (::aluminum_shark::log(::aluminum_shark::AS_DEBUG)) {
    // create logging string
    for (auto& l : arg_literals) {
      ss << l << ", ";
    }
    AS_LOG_DEBUG << "Evaluate: " << std::to_string(arg_literals.size())
                 << " literals: " << ss.str();
    ss.str("");
  } else if (::aluminum_shark::log(::aluminum_shark::AS_INFO)) {
    // create logging string
    for (auto& l : arg_literals) {
      ss << l.shape().ToString() << ", ";
    }
    AS_LOG_INFO << "Evaluate: " << std::to_string(arg_literals.size())
                << " literals: " << ss.str();
  }
  auto ret = evaluator_->Evaluate(computation, arg_literals);
  if (::aluminum_shark::log(::aluminum_shark::AS_DEBUG)) {
    AS_LOG_DEBUG << "computation result:" << ret.ValueOrDie() << std::endl;
  } else if (::aluminum_shark::log(::aluminum_shark::AS_INFO)) {
    AS_LOG_INFO << "computation result:" << ret.ValueOrDie().ToStringOneline()
                << std::endl;
  }

  return ret;
}

/*static*/ int64_t AluminumSharkExecutable::ShapeSizeBytes(const Shape& shape) {
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

}  // namespace aluminum_shark
}  // namespace xla
