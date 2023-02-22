#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LAYOUT_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LAYOUT_H

#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/he_backend/he_backend.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/plugin/aluminum_shark/utils/parallel.h"
#include "tensorflow/compiler/plugin/aluminum_shark/utils/utils.h"
// removes all the depencies to tensorflow, xla and absl. for testing. disables
// all of functions
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"

// #define LAYOUT_DEBUG

namespace aluminum_shark {

enum LAYOUT_TYPE { UNSUPPORTED = -1, SIMPLE, BATCH, E2DM };

extern const std::vector<std::string> LAYOUT_TYPE_STRINGS;
extern const std::vector<const char*> LAYOUT_TYPE_C_STRINGS;

const std::string& layout_type_to_string(LAYOUT_TYPE lt);

const LAYOUT_TYPE string_to_layout_type(const char* name);

/* Layout describes how the slots of one or more ciphertexts or
plaintexts (all of this is true for both plain and ciphertext but for ease
of use we only talk about the ciphertexts from here on out ) map to a message
tensor. The difference between messages and ciphertexts/plaintexts is that
ciphertexts/plaintexts are encoded into polynomials a message is not. We can
think of messages as numpy arrays and the ciphertext is the encoded form.
Internally a message is stored in a contious block of memory with a shape. The
shape allows of multiindexing just like numpy arrays e.g. x[1,2,3] where the
list (1,2,3) is translated into a single index into the storage memory block.
Mapping a message onto one or multiple ciphertexts can change the order of
elements as they appear in the ciphertext. For example, with batch encoding,
the message x with the shape (10,10) would be encoded into in 10 ciphertexts
where the ith ciphertext contains the values x[:,i]. The layout tells us how
to create the mapping from a message to a ciphertext and vice versa.

The mapping from message to ciphertext tells us how we need to arange the
values prior to encoding. It maps from N->N^2. The value at index i in the
flat storage block of the massage goes into kth sloth of the jth ciphertext
according to the mapping. Given the storage of the message x and 2D array y,
where each row will be encoded into single ciphertext, and the mapping
function std::vector<size_t, size_t> map(size_t i) code would look somehting
like this:

for (size_t i = 0; i < x.size(); ++i){
  auto idx = map(i);
  y[idx[0]][idx[1]] = x[i];
}

The reverse lookup maps from N^2->N. Here each value in the mapping tells us
which slot from which ciphertext goes into which index of the message storage
block. Given the storage of the message x and 2D array y, where each row is a
decrypted ciphertext, and the mapping function
std::vector<size_t, size_t> map(std::vector<size_t, size_t> i) code would look
somehting like this:

for (size_t i = 0; i < x.size(); ++i){
  auto idx = map(i);
  x[i] = y[idx[0]][idx[1]];
}


*/

// forward declerattions
class Ctxt;
class Ptxt;

class Layout {
 public:
  Layout(const Shape& shape);

  virtual ~Layout(){};

  // builds the internal data structures. it needs to be called after the
  // object has been constructed. using the createLayout function takes care
  // of that.
  virtual void init() = 0;

  virtual LAYOUT_TYPE type() const = 0;

  virtual Layout* deepCopy() const = 0;

  // const std::vector<size_t>& map(size_t i) { return indicies_[i]; };
  const Shape& shape() const { return shape_; };
  xla::Shape shape_xla() const;
  const size_t size() const { return size_; };

  virtual std::pair<size_t, size_t> get_layout_index(size_t i) const = 0;
  // virtual size_t get_reverse_index(size_t i, size_t j) const = 0;

  template <typename T>
  std::vector<std::vector<T>> layout_vector(const std::vector<T>& vec) const {
    // create return vector
    AS_LOG_INFO << "axis_0_ " << axis_0_ << ", axis_1_ " << axis_1_
                << std::endl;
    std::vector<std::vector<T>> ret_vec(axis_0_, std::vector<T>(axis_1_, 0));
    // copy values into return vector
    AS_LOG_INFO << "laying out vector with size " << vec.size() << " have "
                << indicies_.size() << " indicies" << std::endl;
    AS_LOG_DEBUG << "ret_vec.size() = " << ret_vec.size() << std::endl;
    // create layout function
    std::mutex mu;
    auto func = [this, &ret_vec, &vec, &mu](size_t i) {
      const auto idx = get_layout_index(i);
      const size_t idx_0 = idx.first;
      const size_t idx_1 = idx.second;
      // only lock if we actually need it
      if (log(AS_DEBUG) &&
          false) {  // currently disabled cause it takes forever
        std::unique_lock<std::mutex>(mu);
        AS_LOG_DEBUG << "i" << i << " -> " << idx_0 << " ," << idx_1
                     << std::endl;
        AS_LOG_DEBUG << "ret_vec[" << idx_0
                     << "].size() = " << ret_vec[idx_0].size() << std::endl;
      }
      ret_vec[idx_0][idx_1] = vec[i];
    };

    run_parallel(0, vec.size(), func);
    return ret_vec;
  };

  template <typename T>
  std::vector<T> reverse_layout_vector(
      const std::vector<std::vector<T>>& vec) const {
    // create return vector
    std::vector<T> ret_vec(size_);
    // copy values into return vector
    AS_LOG_S << "reverse layout" << std::endl;
    for (size_t i = 0; i < ret_vec.size(); ++i) {
      const auto idx = get_layout_index(i);
      const size_t idx_0 = idx.first;
      const size_t idx_1 = idx.second;
      if (log(AS_DEBUG)) {
        AS_LOG_DEBUG << "ret[" << i << "] = vec[" << idx_0 << "][" << idx_1
                     << "]" << std::endl;
      }
      ret_vec[i] = vec[idx_0][idx_1];
    }
    return ret_vec;
  };

  virtual bool is_compatbile(Layout* other) {
    return this->type() == other->type();
  };

  virtual bool is_compatbile(Layout& other) {
    return this->type() == other.type();
  };

  // accessing ctxt data
  // returns the actual shape of the underlying buffer
  virtual Shape get_physical_shape() const = 0;
  // returns the actual shape of the underlying buffer as an xla::Shape
  virtual xla::Shape get_physical_shape_xla() const;

  virtual std::shared_ptr<HECtxt> get(absl::Span<const int64_t> index,
                                      Ctxt& ctxt) const;
  virtual std::shared_ptr<HECtxt> get(size_t index, Ctxt& ctxt) const;

  virtual void set(absl::Span<const int64_t> index, Ctxt& ctxt,
                   std::shared_ptr<HECtxt> value) const;
  virtual void set(size_t index, Ctxt& ctxt,
                   std::shared_ptr<HECtxt> value) const;

  // Operation Interface
  virtual void add_in_place(Ctxt& one, const Ctxt& two) const = 0;
  virtual void multiply_in_place(Ctxt& one, const Ctxt& two) const = 0;

  virtual void add_in_place(Ctxt& one, const Ptxt& two) const = 0;
  virtual void multiply_in_place(Ctxt& one, const Ptxt& two) const = 0;

  virtual void add_in_place(Ctxt& one, long two) const = 0;
  virtual void multiply_in_place(Ctxt& one, long two) const = 0;

  virtual void add_in_place(Ctxt& one, double two) const = 0;
  virtual void multiply_in_place(Ctxt& one, double two) const = 0;

  // matrix and vector operations

  // dot is the general entry point
  virtual Ctxt dot(const Ctxt& one, const Ctxt& two) const = 0;
  virtual Ctxt dot(const Ctxt& one, const Ptxt& two) const = 0;
  // Matrix multplication
  virtual Ctxt mat_mult(const Ctxt& one, const Ctxt& two) const = 0;
  virtual Ctxt mat_mult(const Ctxt& one, const Ptxt& two) const = 0;
  // More general matrix multplication for hihger dimensional matrices
  // see: https://www.tensorflow.org/xla/operation_semantics#dotgeneral, and
  // https://en.wikipedia.org/wiki/Tensor_contraction
  virtual Ctxt mat_mult_general(const Ctxt& one, const Ctxt& two) const = 0;
  virtual Ctxt mat_mult_general(const Ctxt& one, const Ptxt& two) const = 0;

  // others

  virtual Ctxt convolution(const Ctxt& lhs, const Ptxt& rhs,
                           xla::HloInstruction* hlo) const = 0;

  virtual Ctxt reshape(Ctxt& lhs, const Shape& shape) const = 0;

  virtual Ctxt pad(Ctxt& lhs, const xla::PaddingConfig& pad_config,
                   const xla::Shape& new_shape, double pad_value) const;

 protected:
  Shape shape_;
  size_t size_;  // number of total elements
  std::vector<std::vector<size_t>> indicies_;
  size_t axis_0_,
      axis_1_;  // number of cipher texts, number of elements in a ciphertext
};

class SimpleLayout : public Layout {
 public:
  SimpleLayout(const Shape& shape) : Layout(shape){};
  virtual void init() override;

  virtual std::pair<size_t, size_t> get_layout_index(size_t i) const;

  virtual LAYOUT_TYPE type() const override;
  virtual Layout* deepCopy() const override;

  // accessing ctxt data
  // returns the actual shape of the underlying buffer
  virtual Shape get_physical_shape() const override;

  // Operation Interface
  virtual void add_in_place(Ctxt& one, const Ctxt& two) const override;
  virtual void multiply_in_place(Ctxt& one, const Ctxt& two) const override;

  virtual void add_in_place(Ctxt& one, const Ptxt& two) const override;
  virtual void multiply_in_place(Ctxt& one, const Ptxt& two) const override;

  virtual void add_in_place(Ctxt& one, long two) const override;
  virtual void multiply_in_place(Ctxt& one, long two) const override;

  virtual void add_in_place(Ctxt& one, double two) const override;
  virtual void multiply_in_place(Ctxt& one, double two) const override;

  // matrix and vector operations

  // dot is the general entry point
  virtual Ctxt dot(const Ctxt& one, const Ptxt& two) const override;
  virtual Ctxt dot(const Ctxt& one, const Ctxt& two) const;
  // Matrix multplication
  virtual Ctxt mat_mult(const Ctxt& one, const Ptxt& two) const override;
  virtual Ctxt mat_mult(const Ctxt& one, const Ctxt& two) const override;
  // More general matrix multplication for hihger dimensional matrices
  // see: https://www.tensorflow.org/xla/operation_semantics#dotgeneral, and
  // https://en.wikipedia.org/wiki/Tensor_contraction
  virtual Ctxt mat_mult_general(const Ctxt& one,
                                const Ptxt& two) const override;
  virtual Ctxt mat_mult_general(const Ctxt& one, const Ctxt& two) const;

  // others
  virtual Ctxt convolution(const Ctxt& lhs, const Ptxt& rhs,
                           xla::HloInstruction* hlo) const override;

  virtual Ctxt reshape(Ctxt& lhs, const Shape& shape) const override;

  virtual Ctxt pad(Ctxt& lhs, const xla::PaddingConfig& pad_config,
                   const xla::Shape& new_shape,
                   double pad_value) const override;

 private:
  template <class T, class U>
  Ctxt dot_internal(const Ctxt& one, const T& two) const;

  template <class T, class U>
  Ctxt mat_mult_internal(const Ctxt& one, const T& two) const;

  template <class T, class U>
  Ctxt mat_mult_general_internal(const Ctxt& one, const T& two) const;
};

class BatchLayout : public Layout {
 public:
  BatchLayout(const Shape& shape) : Layout(shape){};
  virtual void init() override;

  virtual std::pair<size_t, size_t> get_layout_index(size_t i) const;
  virtual LAYOUT_TYPE type() const override;
  virtual Layout* deepCopy() const override;

  // accessing ctxt data
  // returns the actual shape of the underlying buffer
  virtual Shape get_physical_shape() const override;

  // Operation Interface
  virtual void add_in_place(Ctxt& one, const Ctxt& two) const override;
  virtual void multiply_in_place(Ctxt& one, const Ctxt& two) const override;

  virtual void add_in_place(Ctxt& one, const Ptxt& two) const override;
  virtual void multiply_in_place(Ctxt& one, const Ptxt& two) const override;

  virtual void add_in_place(Ctxt& one, long two) const override;
  virtual void multiply_in_place(Ctxt& one, long two) const override;

  virtual void add_in_place(Ctxt& one, double two) const override;
  virtual void multiply_in_place(Ctxt& one, double two) const override;

  // matrix and vector operations

  // Dot product between two vectors
  virtual Ctxt dot(const Ctxt& one, const Ctxt& two) const override;
  virtual Ctxt dot(const Ctxt& one, const Ptxt& two) const override;
  // Matrix multplication
  virtual Ctxt mat_mult(const Ctxt& one, const Ctxt& two) const override;
  virtual Ctxt mat_mult(const Ctxt& one, const Ptxt& two) const override;
  // More general matrix multplication for hihger dimensional matrices
  // see: https://www.tensorflow.org/xla/operation_semantics#dotgeneral, and
  // https://en.wikipedia.org/wiki/Tensor_contraction
  virtual Ctxt mat_mult_general(const Ctxt& one,
                                const Ctxt& two) const override;
  virtual Ctxt mat_mult_general(const Ctxt& one,
                                const Ptxt& two) const override;

  // others

  virtual Ctxt convolution(const Ctxt& lhs, const Ptxt& rhs,
                           xla::HloInstruction* hlo) const override;

  virtual Ctxt reshape(Ctxt& lhs, const Shape& shape) const;

  virtual Ctxt pad(Ctxt& lhs, const xla::PaddingConfig& pad_config,
                   const xla::Shape& new_shape,
                   double pad_value) const override;

 private:
  template <class T, class U>
  Ctxt dot_internal(const Ctxt& one, const T& two) const;

  template <class T, class U>
  Ctxt mat_mult_internal(const Ctxt& one, const T& two) const;
};

Layout* createLayout(const char* type, const Shape& shape);

Layout* createLayout(const LAYOUT_TYPE type, const Shape& shape);

Layout* createLayout(const LAYOUT_TYPE type, const xla::Shape& shape);

// helper functions
xla::Shape create_xla_dummy_shape(const Shape& shape);

Shape xla_shape_to_shark_shape(const xla::Shape& shape);

// helper function that computes low level dot products.
// one and tow should be std::pair that hold the start and
// end iterator to the vectors
template <class T, class U>
std::vector<std::shared_ptr<HECtxt>> simple_dot_helper(
    const std::pair<T, T>& one, const std::pair<U, U>& two) {
  // perform first multiplication
  auto iter_one = one.first;
  auto iter_two = two.first;
  AS_LOG_S << "starting simple dot" << std::endl;
  HECtxt* result = **(one.first) * two.first->get();
  AS_LOG_S << "starting simple dot" << std::endl;
#ifdef LAYOUT_DEBUG
  const HEContext* context = result->getContext();
  AS_LOG_S << "decrypted: " << context->decryptDouble(one.first->get())[0]
           << " * " << context->decryptDouble(two.first->get())[0] << " = "
           << context->decryptDouble(result)[0] << std::endl;
#endif
  // need to increment both iterator;
  ++iter_one;
  ++iter_two;
  size_t i = 0;
  for (; iter_one != one.second; ++iter_one, ++iter_two, i++) {
    // need to create a temporary variable so we can free
    // it later
    AS_LOG_S << "performing product" << std::endl;
    HECtxt* temp = **iter_one * iter_two->get();
#ifdef LAYOUT_DEBUG
    AS_LOG_S << "decrypted: " << context->decryptDouble(iter_one->get())[0]
             << " * " << context->decryptDouble(iter_two->get())[0] << " = "
             << context->decryptDouble(temp)[0] << std::endl;
    AS_LOG_S << "decrypted: " << context->decryptDouble(result)[0] << " + "
             << context->decryptDouble(temp)[0] << " = ";
#endif
    AS_LOG_S << "performing " << i << "th addition" << std::endl;
    result->addInPlace(temp);
#ifdef LAYOUT_DEBUG
    AS_LOG_SA << context->decryptDouble(result)[0] << std::endl;
#endif
    // we have taken ownership of the pointer and are now
    // repsonible for it
    delete temp;
  }
  return std::vector<std::shared_ptr<HECtxt>>{std::shared_ptr<HECtxt>(result)};
}

void registerLayout(LAYOUT_TYPE type,
                    std::function<Layout*(const Shape& shape)> factory);

std::ostream& operator<<(std::ostream& os, const Layout& layout);

template <class T>
size_t size_of_shape(const T& shape) {
  size_t result = 1;
  for (const auto& i : shape) {
    result *= i;
  }
  return result;
}

}  //  namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LAYOUT_H \
        */
