#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LAYOUT_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LAYOUT_H

#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

namespace aluminum_shark {

enum LAYOUT_TYPE { UNSUPPORTED = -1, SIMPLE, BATCH };

extern const std::vector<const char*> LAYOUT_TYPE_STRINGS;

const char* layout_type_to_string(LAYOUT_TYPE lt);

const LAYOUT_TYPE string_to_layout_type(const char* name);

using Shape = std::vector<size_t>;

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
elements as they appear in the ciphertext. For example, with batch encoding, the
message x with the shape (10,10) would be encoded into in 10 ciphertexts where
the ith ciphertext contains the values x[:,i]. The layout tells us how to create
the mapping from a message to a ciphertext and vice versa.

The mapping from message to ciphertext tells us how we need to arange the values
prior to encoding. It maps from N->N^2. The value at index i in the flat storage
block of the massage goes into kth sloth of the jth ciphertext according to the
mapping. Given the storage of the message x and 2D array y, where each row will
be encoded into single ciphertext, and the mapping function
std::vector<size_t, size_t> map(size_t i) code would look somehting like this:

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
  Layout(Shape& shape);

  virtual ~Layout(){};

  // builds the internal data structures. it needs to be called after the object
  // has been constructed. using the createLayout function takes care of that.
  virtual void init() = 0;

  virtual LAYOUT_TYPE type() const = 0;

  virtual Layout* deepCopy() const = 0;

  const std::vector<size_t>& map(size_t i) { return indicies_[i]; };
  const Shape& shape() { return shape_; };

  template <typename T>
  std::vector<std::vector<T>> layout_vector(const std::vector<T>& vec) const {
    // create return vector
    std::vector<std::vector<T>> ret_vec(axis_0_, std::vector<T>(axis_1_, 0));
    // copy values into return vector
    AS_LOG_S << "laying out vector " << vec.size() << std::endl;
    for (size_t i = 0; i < vec.size(); ++i) {
      const auto& idx = indicies_[i];
      AS_LOG_S << "i" << i << " -> " << idx[0] << " ," << idx[1] << std::endl;
      AS_LOG_S << "ret_vec[" << idx[0] << "]"
               << " : " << ret_vec[idx[0]].size() << std::endl;
      ret_vec[idx[0]][idx[1]] = vec[i];
    }
    return ret_vec;
  };

  template <typename T>
  std::vector<T> reverse_layout_vector(
      const std::vector<std::vector<T>>& vec) const {
    // create return vector
    std::vector<T> ret_vec(size_);
    // copy values into return vector
    for (size_t i = 0; i < ret_vec.size(); ++i) {
      const auto& idx = indicies_[i];
      ret_vec[i] = vec[idx[0]][idx[1]];
    }
    return ret_vec;
  };

  virtual bool is_compatbile(Layout* other) {
    return this->type() == other->type();
  };

  virtual bool is_compatbile(Layout& other) {
    return this->type() == other.type();
  };

  // Operation Interface
  virtual void add_in_place(Ctxt& one, const Ctxt& two) const = 0;
  virtual void multiply_in_place(Ctxt& one, const Ctxt& two) const = 0;

  virtual void add_in_place(Ctxt& one, const Ptxt& two) const = 0;
  virtual void multiply_in_place(Ctxt& one, const Ptxt& two) const = 0;

  virtual void add_in_place(Ptxt& one, const Ptxt& two) const = 0;
  virtual void multiply_in_place(Ptxt& one, const Ptxt& two) const = 0;

  virtual void add_in_place(Ctxt& one, long two) const = 0;
  virtual void multiply_in_place(Ctxt& one, long two) const = 0;

  virtual void add_in_place(Ctxt& one, double two) const = 0;
  virtual void multiply_in_place(Ctxt& one, double two) const = 0;

  virtual void add_in_place(Ptxt& one, long two) const = 0;
  virtual void multiply_in_place(Ptxt& one, long two) const = 0;

  virtual void add_in_place(Ptxt& one, double two) const = 0;
  virtual void multiply_in_place(Ptxt& one, double two) const = 0;

 protected:
  Shape shape_;
  size_t size_;
  std::vector<std::vector<size_t>> indicies_;
  size_t axis_0_,
      axis_1_;  // number of cipher texts, number of elements in a ciphertext
};

class SimpleLayout : public Layout {
 public:
  SimpleLayout(Shape& shape) : Layout(shape){};
  virtual void init() override;
  virtual LAYOUT_TYPE type() const override;
  virtual Layout* deepCopy() const override;

  // Operation Interface
  virtual void add_in_place(Ctxt& one, const Ctxt& two) const override;
  virtual void multiply_in_place(Ctxt& one, const Ctxt& two) const override;

  virtual void add_in_place(Ctxt& one, const Ptxt& two) const override;
  virtual void multiply_in_place(Ctxt& one, const Ptxt& two) const override;

  virtual void add_in_place(Ptxt& one, const Ptxt& two) const override;
  virtual void multiply_in_place(Ptxt& one, const Ptxt& two) const override;

  virtual void add_in_place(Ctxt& one, long two) const override;
  virtual void multiply_in_place(Ctxt& one, long two) const override;

  virtual void add_in_place(Ctxt& one, double two) const override;
  virtual void multiply_in_place(Ctxt& one, double two) const override;

  virtual void add_in_place(Ptxt& one, long two) const override;
  virtual void multiply_in_place(Ptxt& one, long two) const override;

  virtual void add_in_place(Ptxt& one, double two) const override;
  virtual void multiply_in_place(Ptxt& one, double two) const override;
};

class BatchLayout : public Layout {
 public:
  BatchLayout(Shape& shape) : Layout(shape){};
  virtual void init() override;
  virtual LAYOUT_TYPE type() const override;
  virtual Layout* deepCopy() const override;

  // Operation Interface
  virtual void add_in_place(Ctxt& one, const Ctxt& two) const override;
  virtual void multiply_in_place(Ctxt& one, const Ctxt& two) const override;

  virtual void add_in_place(Ctxt& one, const Ptxt& two) const override;
  virtual void multiply_in_place(Ctxt& one, const Ptxt& two) const override;

  virtual void add_in_place(Ptxt& one, const Ptxt& two) const override;
  virtual void multiply_in_place(Ptxt& one, const Ptxt& two) const override;

  virtual void add_in_place(Ctxt& one, long two) const override;
  virtual void multiply_in_place(Ctxt& one, long two) const override;

  virtual void add_in_place(Ctxt& one, double two) const override;
  virtual void multiply_in_place(Ctxt& one, double two) const override;

  virtual void add_in_place(Ptxt& one, long two) const override;
  virtual void multiply_in_place(Ptxt& one, long two) const override;

  virtual void add_in_place(Ptxt& one, double two) const override;
  virtual void multiply_in_place(Ptxt& one, double two) const override;
};

Layout* createLayout(const char* type, Shape& shape);

Layout* createLayout(const LAYOUT_TYPE type, Shape& shape);

}  //  namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LAYOUT_H \
        */
