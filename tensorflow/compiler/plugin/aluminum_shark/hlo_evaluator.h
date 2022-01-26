/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_HLO_EVALUATOR_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_HLO_EVALUATOR_H

#define _USE_MATH_DEFINES

#include <functional>
#include <map>
#include <memory>

#include "absl/container/node_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/plugin/aluminum_shark/base_txt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/plugin/aluminum_shark/ptxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/python/python_handle.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/dynamic_dimension_inference.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {
namespace aluminum_shark {

// Responsible for evaluating HLO and obtain literal as the evaluation
// results.
//
// This class is not thread-safe.
class AluminumSharkHloEvaluator : public DfsHloVisitorWithDefault {
 public:
  // Only evaluate up to max_loop_iterations per while-loop execution if
  // specified.
  explicit AluminumSharkHloEvaluator(int64_t max_loop_iterations = -1);

  // Evaluates an HLO module and an array of pointers to literals.  Returns the
  // evaluated result as a literal if successful.
  //
  // Precondition: The indices of arg_literals correspond to the parameter
  // numbers of the HLO parameters in the computation. See comment below for an
  // example.
  //
  // (Dummy template arg is to reduce the overloading priority of one overload
  // so that Evaluate(module, {}) resolves unambiguously.)
  StatusOr<Literal> Evaluate(const HloModule& module,
                             absl::Span<const Literal* const> arg_literals) {
    return Evaluate(*module.entry_computation(), arg_literals);
  }
  template <typename Dummy = void>
  StatusOr<Literal> Evaluate(const HloModule& module,
                             absl::Span<const Literal> arg_literals) {
    return Evaluate(*module.entry_computation(), arg_literals);
  }

  // Evaluates an HLO computation and an array of pointers to literals.
  // Returns the evaluated result as a literal if successful.
  // Precondition: The indices of arg_literals correspond to the parameter
  // numbers of the HLO parameters in the computation. For e.g., consider the
  // following graph:
  //
  //                *
  //            /       \
  //            +     Parameter1
  //        /      \
  //       /        \
  //    Parameter0  Constant
  //
  // where Parameter0 has parameter_number 0 and Parameter1 has parameter_number
  // 1 in this computation. The input literals array will then have its first
  // literal map to Parameter0 and the second map to Parameter1.
  //
  // (Dummy template arg is to reduce the overloading priority of one overload
  // so that Evaluate(module, {}) resolves unambiguously.)
  StatusOr<Literal> Evaluate(const HloComputation& computation,
                             absl::Span<const Literal* const> arg_literals);
  template <typename Dummy = void>
  StatusOr<Literal> Evaluate(const HloComputation& computation,
                             absl::Span<const Literal> arg_literals) {
    std::vector<const Literal*> arg_literal_ptrs;

    // take the ctxt passed in from python and map them to literals
    AS_LOG("Creating Ctxts for the input paramets");
    ::aluminum_shark::PythonHandle& ph =
        ::aluminum_shark::PythonHandle::getInstance();
    const std::vector<::aluminum_shark::Ctxt>& ctxts =
        ph.getCurrentCiphertexts();
    int i = 0;
    AS_LOG("Number of arg_literals: " + std::to_string(arg_literals.size()) +
           " number of arg_ctxts: " + std::to_string(ctxts.size()));
    for (const auto& l : arg_literals) {
      arg_literal_ptrs.push_back(&l);

      // create copies the input ctxts
      ::aluminum_shark::Ctxt ctxt = ctxts[i++];
      AS_LOG("Input Literal to Ctxt: " + l.ToStringOneline() + " -> " +
             ctxt.getName());
      arg_ctxts_.push_back(std::move(ctxt));
    }

    return Evaluate(computation, arg_literal_ptrs);
  }

  // Gets the value of running a single HLO instruction.
  //
  // All of the operands to this instruction must be constants.
  StatusOr<Literal> Evaluate(HloInstruction* instruction);

  // Same as Evaluate, except returning false on error and accepts an output
  // pointer.
  bool TryEvaluate(HloInstruction* instruction, Literal* result);

  // Evaluates a single HLO instruction, substituting the given literals for
  // some of the instruction's operands.
  //
  // For example, given instruction = op(A, B, C) and the map
  // {A = x, C = y}, this evaluates op(x, B, y).
  StatusOr<Literal> EvaluateWithSubstitutions(
      const HloInstruction* instruction,
      const std::unordered_map<const HloInstruction*, const Literal*>&
          substitutions);

  StatusOr<Literal> EvaluateElementwiseBinaryOp(HloOpcode opcode,
                                                const Literal& lhs,
                                                const Literal& rhs);

  StatusOr<Literal> EvaluateElementwiseUnaryOp(HloOpcode opcode,
                                               const Literal& operand);

  StatusOr<Literal> EvaluateElementwiseTernaryOp(HloOpcode opcode,
                                                 const Literal& lhs,
                                                 const Literal& rhs,
                                                 const Literal& ehs);

  StatusOr<Literal> EvaluateElementwiseCompareOp(ComparisonDirection direction,
                                                 const Literal& lhs,
                                                 const Literal& rhs);

  StatusOr<Literal> EvaluateDotOp(const DotDimensionNumbers& dim_numbers,
                                  const PrecisionConfig& precision_config,
                                  const Literal& lhs, const Literal& rhs);

  void set_dynamic_dimension_inference(
      DynamicDimensionInference* dynamic_dimension_inference) {
    dynamic_dimension_inference_ = dynamic_dimension_inference;
  }

  DynamicDimensionInference* dynamic_dimension_inference() {
    return dynamic_dimension_inference_;
  }

  // Enable the fast path for certain operations like dot or convolution.
  void set_use_fast_path(bool value) { use_fast_path_ = value; }

  // Handles evaluation of a custom-call op.
  // Operand literals are provided in |operands| and implementations must
  // populate |output| before returning.
  using CustomCallHandler = std::function<StatusOr<Literal>(
      HloInstruction* custom_call, absl::Span<const Literal*> operands)>;

  // Sets a handler that is called during evaluation for custom-call ops.
  // If no handler is defined the default error behavior will occur. The handler
  // will be provided evaluated literals for all operands and is expected to
  // return an output literal of the appropriate shape.
  void set_custom_call_handler(
      std::function<StatusOr<Literal>(HloInstruction* custom_call,
                                      absl::Span<const Literal*> operands)>
          handler) {
    custom_call_handler_ = std::move(handler);
  }

  // Returns the result of a matrix multiply `lhs x rhs`.
  static std::unique_ptr<Array2D<Eigen::half>> MatmulArray2D(
      const Array2D<Eigen::half>& lhs, const Array2D<Eigen::half>& rhs);
  static std::unique_ptr<Array2D<float>> MatmulArray2D(
      const Array2D<float>& lhs, const Array2D<float>& rhs);
  static std::unique_ptr<Array2D<double>> MatmulArray2D(
      const Array2D<double>& lhs, const Array2D<double>& rhs);
  static std::unique_ptr<Array2D<std::complex<float>>> MatmulArray2D(
      const Array2D<std::complex<float>>& lhs,
      const Array2D<std::complex<float>>& rhs);
  static std::unique_ptr<Array2D<std::complex<double>>> MatmulArray2D(
      const Array2D<std::complex<double>>& lhs,
      const Array2D<std::complex<double>>& rhs);
  static std::unique_ptr<Array2D<int32>> MatmulArray2D(
      const Array2D<int32>& lhs, const Array2D<int32>& rhs);

 protected:
  // Make AluminumSharkHloEvaluatorTypedVisitor a friend because it is logically
  // part of this class.
  //
  // A straightforward implementation would be to make it a nested class
  // declared and defined in hlo_evaluator.cc.  Instead
  // AluminumSharkHloEvaluatorTypedVisitor lives as a separate class with its
  // own header because its template gets instantiated many times and we want to
  // use extern templates to shard out the compilation of those instantiations
  // across multiple cc files.
  template <typename ReturnT, typename ElementwiseT>
  friend class AluminumSharkHloEvaluatorTypedVisitor;

  // Wraps around instruction handling to infer types before dispatching to
  // the corresponding typed Visitor.
  Status DefaultAction(HloInstruction* hlo) override {
    return hlo->Visit(typed_visitors_[hlo->shape().element_type()].get());
  }

  Status Preprocess(HloInstruction* hlo) override;

  Status Postprocess(HloInstruction* hlo) override;

  // Operations that are type-agnostic or always return a specific type, such as
  // HandleIsFinite where boolean is always returned.
  //
  Status HandleBitcast(HloInstruction* bitcast) override;

  Status HandleGetDimensionSize(HloInstruction* get_dimension_size) override;

  Status HandleSetDimensionSize(HloInstruction* set_dimension_size) override;

  Status HandleParameter(HloInstruction* parameter) override;

  Status HandleConstant(HloInstruction* constant) override;

  Status HandleConcatenate(HloInstruction* concatenate) override;

  Status HandleReshape(HloInstruction* reshape) override;

  Status HandleTranspose(HloInstruction* transpose) override;

  Status HandleIsFinite(HloInstruction* is_finite) override;

  Status HandleCompare(HloInstruction* compare) override;

  Status HandleTuple(HloInstruction* tuple) override;

  Status HandleFft(HloInstruction* fft) override;

  Status HandleGather(HloInstruction* gather) override;

  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;

  Status HandleCopy(HloInstruction* copy) override;

  Status HandleCopyStart(HloInstruction* copy_start) override;

  Status HandleCopyDone(HloInstruction* copy_done) override;

  Status HandleConditional(HloInstruction* conditional) override;

  Status HandleCall(HloInstruction* call) override;

  Status HandleFusion(HloInstruction* fusion) override;

  Status HandleWhile(HloInstruction* while_hlo) override;

  Status HandleSelect(HloInstruction* select) override;

  Status HandleTupleSelect(HloInstruction* tuple_select) override;

  Status HandleBroadcast(HloInstruction* broadcast) override;

  Status HandleAfterAll(HloInstruction* after_all) override;

  Status HandleAddDependency(HloInstruction* add_dependency) override;

  Status HandleSort(HloInstruction* sort) override;

  Status HandleReal(HloInstruction* real) override;

  Status HandleImag(HloInstruction* imag) override;

  Status HandleComplex(HloInstruction* complex) override;

  Status HandleReduce(HloInstruction* reduce) override;

  Status HandleReduceWindow(HloInstruction* hlo) override;

  Status HandleCustomCall(HloInstruction* custom_call) override;

  // Unsupported HLOs, note some of them (such as BatchNorm*) are typically
  // expanded in a semantic-preserving way into other HLOs by adding expansion
  // HLO pass to the HLO optimization pass during compilation, which can then be
  // handled by the evaluator.
  Status HandleBatchNormGrad(HloInstruction* batch_norm_grad) override {
    return Unimplemented("BatchNormGrad HLO is unsupported by the evaluator.");
  };
  Status HandleBatchNormInference(
      HloInstruction* batch_norm_inference) override {
    return Unimplemented(
        "BatchNormInference HLO is unsupported by the evaluator.");
  };
  Status HandleBatchNormTraining(HloInstruction* batch_norm_training) override {
    return Unimplemented(
        "BatchNormTraining HLO is unsupported by the evaluator.");
  };
  Status HandleInfeed(HloInstruction* infeed) override {
    return Unimplemented("Infeed HLO is unsupported by the evaluator.");
  };
  Status HandleOutfeed(HloInstruction* outfeed) override {
    return Unimplemented("Outfeed HLO is unsupported by the evaluator.");
  };

  // Returns the already-evaluated literal result for the instruction.
  //
  // A Constant instruction is considered evaluated and its literal will be
  // returned directly without looking up the cache.
  //
  // Similarly, a Parameter instruction is considered evaluated and its literal
  // is looked up in arg_literals.
  //
  // Crash with log if the given instruction has not been evaluated previously.
  const Literal& GetEvaluatedLiteralFor(const HloInstruction* hlo) {
    if (hlo->IsConstant()) {
      return hlo->literal();
    }
    if (hlo->opcode() == HloOpcode::kParameter) {
      return *arg_literals_.at(hlo->parameter_number());
    }
    auto it = evaluated_.find(hlo);
    CHECK(it != evaluated_.end())
        << "could not find evaluated value for: " << hlo->ToString();
    return it->second;
  }

  // Tracks the HLO instruction and its evaluated literal result.
  //
  // Parameters and constants aren't stored here, see implementation of
  // GetEvaluatedLiteralFor.
  //
  // TODO(b/35950897): have better memory management here to free instructions
  // that are no longer a parent for any other subsequent instruction in
  // post-ordering.
  //
  // Must be cleared for each evaluation.
  //
  // Storing Literal in place requires the container to have pointer stability
  // so we cannot use flat_hash_map any more.
  absl::node_hash_map<const HloInstruction*, Literal> evaluated_;

  // Ctxt stuff
  //
  // We need to keep track of the Ctxts just like the literals above. Once an
  // HloInstruction has been evaluted we store the resulting Ctxt in this map.
  // Ctxt can be quite memory intensive so we don't want to copy them around.
  // For now the evaluator owns the Ctxt. Once computation is complete we pass a
  // pointer to the outside. This could be a problem in terms of lifetime, if
  // the evaluator goes out of scope. In that case we need have the Ctxt be
  // owned by a nother object with a longer lifetime.
  //
  // this `GetEvaluatedCtxtFor` can either reutrn a Ctxt or a Ptxt therefore it
  // returns the base type `BaseTxt`

  // Just like the literals we are tracking Ctxt to HLO mapping
  ::aluminum_shark::BaseTxt& GetEvaluatedCtxtFor(const HloInstruction* hlo) {
    AS_LOG("Getting Ctxt for " + hlo->name());
    if (hlo->IsConstant()) {
      auto& literal = hlo->literal();
      AS_LOG("Converting constant to ctxt, literal: " + literal.ToString() +
             " Hlo shape: " + hlo->shape().ToString());
      if (hlo->shape().rank() == 0) {
        // get the context that we are working with by taking it from the first
        // input parameter
        AS_LOG("Getting Ctxt");
        ::aluminum_shark::Ctxt& tempctxt = arg_ctxts_.at(0);
        AS_LOG("Getting HECtxt");
        ::aluminum_shark::HECtxt& hectxt = tempctxt.getValue();
        AS_LOG("Getting Context");
        const ::aluminum_shark::HEContext* context = hectxt.getContext();
        if (::xla::primitive_util::IsFloatingPointType(
                literal.shape().element_type())) {
          AS_LOG("Converting to float");
          double value = literal.GetAsDouble({}).value();
          contant_ptxt_[hlo] = ::aluminum_shark::Ptxt(
              context->encode(std::vector<double>{value}), hlo->name());
          AS_LOG("Convertted constant to ptxt: " +
                 contant_ptxt_[hlo].to_string());
        } else {
          AS_LOG("Converting to long");
          long value = literal.GetIntegralAsS64({}).value();
          AS_LOG_S << "Context " << reinterpret_cast<const void*>(context)
                   << std::endl;
          AS_LOG_S << "Context info " << context->to_string() << std::endl;
          contant_ptxt_[hlo] = ::aluminum_shark::Ptxt(
              context->encode(std::vector<long>{value}), hlo->name());
          AS_LOG("Convertted constant to ptxt: " +
                 contant_ptxt_[hlo].to_string());
        }
      } else {
        // TODO: handle none scalar literals
        AS_LOG("Only scalar constanst are supported at the moment");
      }
      return contant_ptxt_[hlo];
    }
    if (hlo->opcode() == HloOpcode::kParameter) {
      return arg_ctxts_.at(hlo->parameter_number());
    }
    AS_LOG_S << "Searching through maps " << std::endl;
    auto it = evaluated_ctxt_.find(hlo);
    // it could be that we are actually dealing with a plaintext
    if (it == evaluated_ctxt_.end()) {
      AS_LOG_S << "Did not find a ctxt for " << hlo->ToString()
               << " looking for a ptxt" << std::endl;
      auto it_ptxt = contant_ptxt_.find(hlo);
      if (it_ptxt != contant_ptxt_.end()) {
        return it_ptxt->second;
      }
      AS_LOG_S << "Did not find a ptxt for " << hlo->ToString() << " either"
               << std::endl;
      AS_LOG_S << "Dumping maps: " << std::endl;
      for (auto iter = evaluated_ctxt_.begin(); iter != evaluated_ctxt_.end();
           iter++) {
        AS_LOG_SA << "\t" << iter->first->ToString() << std::endl;
      }
      for (auto iter = contant_ptxt_.begin(); iter != contant_ptxt_.end();
           iter++) {
        AS_LOG_SA << "\t" << iter->first->ToString() << std::endl;
      }
    }
    CHECK(it != evaluated_ctxt_.end())
        << "could not find evaluated value for: " << hlo->ToString();
    return it->second;
  }

  // helper function to assign plaintexts and ciphertexts to correct storage
  // objects
  void unwrapBaseTxt(const HloInstruction* hlo,
                     ::aluminum_shark::BaseTxt& base) {
    try {
      evaluated_ctxt_[hlo] = dynamic_cast<::aluminum_shark::Ctxt&>(base);
    } catch (const std::bad_cast& e) {
      // that's ok. ignore
    }
    contant_ptxt_[hlo] = static_cast<::aluminum_shark::Ptxt&>(base);
  }

  void unwrapBaseTxt(const HloInstruction* hlo,
                     std::shared_ptr<::aluminum_shark::BaseTxt> base) {
    unwrapBaseTxt(hlo, *base);
  }

  // TODO: think about one structure for both. it would make the lookup and
  // unwrapping easier. like `std::map<const HloInstruction*, BaseTxt>`

  // mapping for evaluated Ctxt. for now this structure owns the Ctxt
  std::map<const HloInstruction*, ::aluminum_shark::Ctxt> evaluated_ctxt_;

  // mapping for ptxt constants
  std::map<const HloInstruction*, ::aluminum_shark::Ptxt> contant_ptxt_;

  // we deal with the input parameters in the same way the plaintexts are dealt
  // with
  std::vector<::aluminum_shark::Ctxt> arg_ctxts_;

  // Use fast path that uses eigen in the evaluator.
  bool use_fast_path_ = false;

 private:
  template <typename ReturnT, typename NativeT>
  static StatusOr<Literal> ElementWiseUnaryOpImpl(
      HloInstruction* instruction,
      const std::function<ReturnT(NativeT)>& unary_op,
      const Literal& operand_literal) {
    const auto shape = instruction->shape();
    const auto* operand = instruction->operand(0);
    TF_RET_CHECK(ShapeUtil::SameDimensions(shape, operand->shape()));

    Literal result(shape);
    TF_RETURN_IF_ERROR(
        result.Populate<ReturnT>([&](absl::Span<const int64_t> multi_index) {
          return unary_op(operand_literal.Get<NativeT>(multi_index));
        }));
    return std::move(result);
  }

  // Map from a primitive type to its associated (templated) DfsHloVisitor.
  std::unique_ptr<DfsHloVisitor> typed_visitors_[PrimitiveType_ARRAYSIZE];

  // Caches pointers to input literals, assuming they are in post-order.
  // Literals are not owned by this class, and they must outlive the lifetime of
  // each invocation to the Evaluate* method.
  // Must be cleared for each evaluation.
  std::vector<const Literal*> arg_literals_;

  // Max loop iterations to execute with no maximum if negative.
  int64_t max_loop_iterations_ = 0;

  // Module-level seed handle.
  uint64 seed_ = 0;
  // RNG engine.
  std::minstd_rand0 engine_;

  // DynamicDimensionInference is used to evaluate GetDimensionSize, which
  // returns the dynamic dimension size of its operand.
  DynamicDimensionInference* dynamic_dimension_inference_ = nullptr;

  // Optional handler for custom_call ops.
  std::function<StatusOr<Literal>(HloInstruction* custom_call,
                                  absl::Span<const Literal*> operands)>
      custom_call_handler_;

  TF_DISALLOW_COPY_AND_ASSIGN(AluminumSharkHloEvaluator);

};  // namespace DfsHloVisitorWithDefault

std::unique_ptr<Array2D<float>> MatmulArray2D(const Array2D<float>& lhs,
                                              const Array2D<float>& rhs);

}  // namespace aluminum_shark
}  // namespace xla

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_HLO_EVALUATOR_H \
        */
