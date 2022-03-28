#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LAYOUT_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LAYOUT_H

#include <string>
#include <vector>

namespace aluminum_shark {

using Shape = std::vector<size_t>;
using Index = std::vector<size_t>;

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

class Layout {
 public:
  Layout(Shape& shape) : shape_(shape) {
    size_t size = 1;
    for (auto& i : shape) {
      size *= i;
    }
    size_ = size;
    indicies_.reserve(size_);
    // creates the sub class indicies
    init();
  };

  virtual ~Layout(){};

  // builds the internal data structures
  virtual void init() = 0;

  const std::vector<size_t>& map(size_t i) { return indicies_[i]; };
  const Shape& shape() { return shape_; };

 protected:
  Shape shape_;
  size_t size_;
  std::vector<std::vector<size_t>> indicies_;
};

class SimpleLayout : public Layout {
  SimpleLayout(Shape& shape) : Layout(shape){};
  virtual void init() override;
};

class BatchLayout : public Layout {
  BatchLayout(Shape& shape) : Layout(shape){};
  virtual void init() override;
};

}  //  namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_LAYOUT_H \
        */
