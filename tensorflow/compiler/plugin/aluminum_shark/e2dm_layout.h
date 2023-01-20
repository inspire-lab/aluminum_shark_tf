#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_E2DM_LAYOUT_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_E2DM_LAYOUT_H

#include "tensorflow/compiler/plugin/aluminum_shark/layout.h"

namespace aluminum_shark {

class E2DMLayout : public SimpleLayout {
 public:
  E2DMLayout(const Shape& shape) : SimpleLayout(shape){};
  virtual void init() override;

  virtual std::pair<size_t, size_t> get_layout_index(size_t i) const;

  virtual LAYOUT_TYPE type() const override;
  virtual Layout* deepCopy() const override;

  // Operation Interface
  // handled by Simplelayout

  // matrix and vector operations

  // dot is the general entry point
  virtual Ctxt dot(const Ctxt& one, const Ptxt& two) const override;
  virtual Ctxt dot(const Ctxt& one, const Ctxt& two) const override;
  // Matrix multplication
  virtual Ctxt mat_mult(const Ctxt& one, const Ptxt& two) const override;
  virtual Ctxt mat_mult(const Ctxt& one, const Ctxt& two) const override;
  // More general matrix multplication for hihger dimensional matrices
  // see: https://www.tensorflow.org/xla/operation_semantics#dotgeneral, and
  // https://en.wikipedia.org/wiki/Tensor_contraction
  virtual Ctxt mat_mult_general(const Ctxt& one,
                                const Ptxt& two) const override;
  virtual Ctxt mat_mult_general(const Ctxt& one, const Ctxt& two) const;

  virtual Ctxt convolution(const Ctxt& lhs, const Ptxt& rhs,
                           xla::HloInstruction* hlo) const override;

  virtual Ctxt reshape(Ctxt& lhs, const Shape& shape) const override;

  //  private:
  //   template <class T, class U>
  //   Ctxt dot_internal(const Ctxt& one, const T& two) const;

  //   template <class T, class U>
  //   Ctxt mat_mult_internal(const Ctxt& one, const T& two) const;

  //   template <class T, class U>
  //   Ctxt mat_mult_general_internal(const Ctxt& one, const T& two) const;
};

}  // namespace aluminum_shark
#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_E2DM_LAYOUT_H \
        */
