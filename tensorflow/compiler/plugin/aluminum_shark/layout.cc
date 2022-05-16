#include "tensorflow/compiler/plugin/aluminum_shark/layout.h"

#include <cstring>

#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/plugin/aluminum_shark/ptxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/utils/utils.h"

namespace aluminum_shark {

const std::vector<const char*> LAYOUT_TYPE_STRINGS{"simple", "batch"};

const char* layout_type_to_string(LAYOUT_TYPE lt) {
  if (lt == LAYOUT_TYPE::UNSUPPORTED) {
    return "unsupported";
  }
  return LAYOUT_TYPE_STRINGS[lt];
}

const LAYOUT_TYPE string_to_layout_type(const char* name) {
  for (size_t i = 0; i < LAYOUT_TYPE_STRINGS.size(); ++i) {
    if (strcmp(name, LAYOUT_TYPE_STRINGS[i]) == 0) {
      return static_cast<LAYOUT_TYPE>(i);
    }
  }
  return LAYOUT_TYPE::UNSUPPORTED;
}

// Base

Layout::Layout(Shape& shape) : shape_(shape) {
  size_t size = 1;
  for (auto& i : shape) {
    size *= i;
  }
  size_ = size;
  indicies_.reserve(size_);
  AS_LOG_S << "nubmer of indices " << size << std::endl;
}

// Simple Layout

void SimpleLayout::init() {
  for (size_t i = 0; i < size_; ++i) {
    indicies_.push_back(std::vector<size_t>{i, 0});
  }
  axis_0_ = size_;
  axis_1_ = 1;
  AS_LOG_S << "Created layout indices " << indicies_.size() << std::endl;
  for (const auto& v : indicies_) {
    stream_vector(v);
    AS_LOG_SA << std::endl;
  }
}

LAYOUT_TYPE SimpleLayout::type() const { return LAYOUT_TYPE::SIMPLE; }

Layout* SimpleLayout::deepCopy() const {
  AS_LOG_S << "creating deepcopy of SimpleLayout" << std::endl;
  return new SimpleLayout(*this);
}

// Operation Interface
void SimpleLayout::add_in_place(Ctxt& one, const Ctxt& two) const {
  auto& one_v = one.getValue();
  const auto& two_v = two.getValue();
  AS_LOG_S << "simple layout add in place, value sizes: " << one_v.size()
           << " += " << two_v.size() << std::endl;

  for (size_t i = 0; i < size_; ++i) {
    one_v[i]->addInPlace(two_v[i].get());
  }
  AS_LOG_S << "add in place done " << std::endl;
}

void SimpleLayout::multiply_in_place(Ctxt& one, const Ctxt& two) const {
  AS_LOG_S << "multiplying " << one.getName() << ", " << two.getName()
           << std::endl;
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->multInPlace(two.getValue()[i].get());
  }
}

void SimpleLayout::add_in_place(Ctxt& one, const Ptxt& two) const {
  // TODO: make sure they are in the same layout. if not we need to layout two
  // on the fly
  auto& one_v = one.getValue();
  const auto& two_v = two.getValue();
  AS_LOG_S << "simple layout add in place, value sizes: " << one_v.size()
           << " and " << two_v.size() << std::endl;
  for (size_t i = 0; i < size_; ++i) {
    AS_LOG_S << "ctxt: " << one_v[i]->to_string() << std::endl;
    AS_LOG_S << "ptxt: " << two_v[i]->to_string() << std::endl;
    one_v[i]->addInPlace(two_v[i].get());
  }
  AS_LOG_S << "add in place done " << std::endl;
}

void SimpleLayout::multiply_in_place(Ctxt& one, const Ptxt& two) const {
  // TODO: make sure they are in the same layout. if not we need to layout two
  // on the fly
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->multInPlace(two.getValue()[i].get());
  }
}

void SimpleLayout::add_in_place(Ptxt& one, const Ptxt& two) const {
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->addInPlace(two.getValue()[i].get());
  }
}

void SimpleLayout::multiply_in_place(Ptxt& one, const Ptxt& two) const {
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->multInPlace(two.getValue()[i].get());
  }
}

void SimpleLayout::add_in_place(Ctxt& one, long two) const {
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->addInPlace(two);
  }
}

void SimpleLayout::multiply_in_place(Ctxt& one, long two) const {
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->multInPlace(two);
  }
}

void SimpleLayout::add_in_place(Ctxt& one, double two) const {
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->addInPlace(two);
  }
}

void SimpleLayout::multiply_in_place(Ctxt& one, double two) const {
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->multInPlace(two);
  }
}

void SimpleLayout::add_in_place(Ptxt& one, long two) const {
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->addInPlace(two);
  }
}

void SimpleLayout::multiply_in_place(Ptxt& one, long two) const {
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->multInPlace(two);
  }
}

void SimpleLayout::add_in_place(Ptxt& one, double two) const {
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->addInPlace(two);
  }
}

void SimpleLayout::multiply_in_place(Ptxt& one, double two) const {
  for (size_t i = 0; i < size_; ++i) {
    one.getValue()[i]->multInPlace(two);
  }
}

// Batch Layout

void BatchLayout::init() {
  size_t bs = shape_[0];  // assumes batch dim is first
  size_t step_size = size_ / bs;
  for (size_t i = 0; i < size_; ++i) {
    // put every batch dimension into a single ciphertext
    indicies_.push_back(std::vector<size_t>{i % step_size, i / step_size});
  }
  axis_0_ = bs;
  axis_1_ = step_size;
}

LAYOUT_TYPE BatchLayout::type() const { return LAYOUT_TYPE::BATCH; }

Layout* BatchLayout::deepCopy() const { return new BatchLayout(*this); }

// Operation Interface
void BatchLayout::add_in_place(Ctxt& one, const Ctxt& two) const {
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->addInPlace(two.getValue()[i].get());
  }
}

void BatchLayout::multiply_in_place(Ctxt& one, const Ctxt& two) const {
  auto& one_v = one.getValue();
  const auto& two_v = two.getValue();
  AS_LOG_S << "simple layout multiply in place, value sizes: " << one_v.size()
           << " and " << two_v.size() << std::endl;
  for (size_t i = 0; i < axis_1_; ++i) {
    AS_LOG_S << "ctxt one: " << one_v[i]->to_string() << std::endl;
    AS_LOG_S << "ctxt two: " << two_v[i]->to_string() << std::endl;
    one_v[i]->multInPlace(two_v[i].get());
  }
  AS_LOG_S << "multiplying done " << std::endl;
}

void BatchLayout::add_in_place(Ctxt& one, const Ptxt& two) const {
  // TODO: make sure they are in the same layout. if not we need to layout two
  // on the fly
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->addInPlace(two.getValue()[i].get());
  }
}

void BatchLayout::multiply_in_place(Ctxt& one, const Ptxt& two) const {
  // TODO: make sure they are in the same layout. if not we need to layout two
  // on the fly
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->multInPlace(two.getValue()[i].get());
  }
}

void BatchLayout::add_in_place(Ptxt& one, const Ptxt& two) const {
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->addInPlace(two.getValue()[i].get());
  }
}

void BatchLayout::multiply_in_place(Ptxt& one, const Ptxt& two) const {
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->multInPlace(two.getValue()[i].get());
  }
}

void BatchLayout::add_in_place(Ctxt& one, long two) const {
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->addInPlace(two);
  }
}

void BatchLayout::multiply_in_place(Ctxt& one, long two) const {
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->multInPlace(two);
  }
}

void BatchLayout::add_in_place(Ctxt& one, double two) const {
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->addInPlace(two);
  }
}

void BatchLayout::multiply_in_place(Ctxt& one, double two) const {
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->multInPlace(two);
  }
}

void BatchLayout::add_in_place(Ptxt& one, long two) const {
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->addInPlace(two);
  }
}

void BatchLayout::multiply_in_place(Ptxt& one, long two) const {
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->multInPlace(two);
  }
}

void BatchLayout::add_in_place(Ptxt& one, double two) const {
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->addInPlace(two);
  }
}

void BatchLayout::multiply_in_place(Ptxt& one, double two) const {
  for (size_t i = 0; i < axis_1_; ++i) {
    one.getValue()[i]->multInPlace(two);
  }
}

// Free functions

Layout* createLayout(const char* type, Shape& shape) {
  LAYOUT_TYPE lt = string_to_layout_type(type);
  return createLayout(lt, shape);
}

Layout* createLayout(const LAYOUT_TYPE type, Shape& shape) {
  Layout* layout;
  switch (type) {
    case LAYOUT_TYPE::BATCH:
      layout = new BatchLayout(shape);
      break;
    case LAYOUT_TYPE::UNSUPPORTED:
      AS_LOG_S << "unsupported layout \"" << type
               << "\" passed. Falling back to simple layout" << std::endl;
    default:
      layout = new SimpleLayout(shape);
      break;
  }
  layout->init();
  return layout;
}

}  // namespace aluminum_shark