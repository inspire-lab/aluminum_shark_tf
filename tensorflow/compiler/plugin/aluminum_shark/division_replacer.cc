#include "tensorflow/compiler/plugin/aluminum_shark/division_replacer.h"

#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace aluminum_shark {

bool DivisionReplacer::InstructionMatchesPattern(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kDivide;
}

StatusOr<HloInstruction*> DivisionReplacer::ExpandInstruction(
    HloInstruction* instruction) {
  HloComputation* computation = instruction->parent();
  HloInstruction* lhs = instruction->operands()[0];
  HloInstruction* rhs = instruction->operands()[1];
  Shape shape = lhs->shape();
  AS_LOG_INFO << "running division replacement. replacing "
              << instruction->ToString() << std::endl;

  // create a one
  auto one_value = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::One(shape.element_type())));
  AS_LOG_DEBUG << "created " << one_value->ToString() << std::endl;
  // broad cast it to the correct shape
  Shape rhs_shape = rhs->shape();
  one_value = computation->AddInstruction(HloInstruction::CreateBroadcast(
      rhs_shape, one_value, one_value->shape().dimensions()));
  AS_LOG_DEBUG << "created " << one_value->ToString() << std::endl;
  // divide one_value by rhs
  auto inverse = computation->AddInstruction(HloInstruction::CreateBinary(
      rhs_shape, HloOpcode::kDivide, one_value, rhs));
  AS_LOG_DEBUG << "inverse " << one_value->ToString() << std::endl;
  // create the multiplication
  auto muliplicaton = computation->AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, lhs, inverse));
  AS_LOG_DEBUG << "muliplicaton " << one_value->ToString() << std::endl;

  AS_LOG_INFO << "division replacement complete " << muliplicaton->ToString()
              << std::endl;

  return muliplicaton;
}

}  // namespace aluminum_shark
}  // namespace xla