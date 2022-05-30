#include "tensorflow/compiler/plugin/aluminum_shark/layout.h"

#include <cstring>
#include <stdexcept>
#include <utility>

#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/he_backend/he_backend.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/plugin/aluminum_shark/ptxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/utils/exception.h"
#include "tensorflow/compiler/plugin/aluminum_shark/utils/utils.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

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

Layout::Layout(const Shape& shape) : shape_(shape) {
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
    if (log()) {
      stream_vector(v);
    }
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

// matrix and vector ops

Ctxt SimpleLayout::dot(const Ctxt& one, const Ctxt& two) const {
  return dot_internal<Ctxt, HECtxt>(one, two);
}
Ctxt SimpleLayout::dot(const Ctxt& one, const Ptxt& two) const {
  return dot_internal<Ptxt, HEPtxt>(one, two);
}

// Matrix multplication
Ctxt SimpleLayout::mat_mult(const Ctxt& one, const Ctxt& two) const {
  return mat_mult_internal<Ctxt, HECtxt>(one, two);
}
Ctxt SimpleLayout::mat_mult(const Ctxt& one, const Ptxt& two) const {
  return mat_mult_internal<Ptxt, HEPtxt>(one, two);
}
// More general matrix multplication for hihger dimensional matrices
// see: https://www.tensorflow.org/xla/operation_semantics#dotgeneral, and
// https://en.wikipedia.org/wiki/Tensor_contraction
Ctxt SimpleLayout::mat_mult_general(const Ctxt& one, const Ctxt& two) const {
  return mat_mult_general_internal<Ctxt, HECtxt>(one, two);
}
Ctxt SimpleLayout::mat_mult_general(const Ctxt& one, const Ptxt& two) const {
  return mat_mult_general_internal<Ptxt, HEPtxt>(one, two);
}

template <class T, class U>
Ctxt SimpleLayout::dot_internal(const Ctxt& one, const T& two) const {
  // shape checks
  if (one.shape().size() != 1 || two.shape().size() != 1) {
    // not a vector. run mat mult
    return mat_mult_internal<T, U>(one, two);
  }
  // incompatible vector shapes
  if (one.shape()[0] != two.shape()[0]) {
    AS_LOG_S << "invalid shapes for dot product: ";
    if (log()) {
      stream_vector(one.shape());
    }
    AS_LOG_SA << " and ";
    if (log()) {
      stream_vector(two.shape());
    }
    AS_LOG_SA << std::endl;
    throw std::invalid_argument("shapes incompatible");
  }
  // "compute" resultshape
  Shape result_shape{1};
  // create result layout
  std::shared_ptr<Layout> result_layout(
      createLayout(LAYOUT_TYPE::SIMPLE, result_shape));
  const auto& one_v = one.getValue();
  const auto& two_v = two.getValue();

  // stick begining and end iterators into a pair
  auto one_iters = std::make_pair<
      typename std::vector<std::shared_ptr<HECtxt>>::const_iterator,
      typename std::vector<std::shared_ptr<HECtxt>>::const_iterator>(
      one_v.cbegin(), one_v.cend());
  auto two_iters =
      std::make_pair<typename std::vector<std::shared_ptr<U>>::const_iterator,
                     typename std::vector<std::shared_ptr<U>>::const_iterator>(
          two_v.cbegin(), two_v.cend());
  auto result_ctxts = simple_dot_helper<
      typename std::vector<std::shared_ptr<HECtxt>>::const_iterator,
      typename std::vector<std::shared_ptr<U>>::const_iterator>(one_iters,
                                                                two_iters);
  std::stringstream result_name;
  result_name << one.getName() << " dot " << two.getName();
  return Ctxt(result_ctxts, result_layout, result_name.str());
}

template <class T, class U>
Ctxt SimpleLayout::mat_mult_internal(const Ctxt& one, const T& two) const {
  // shape checks
  // this only works for iif we have 2 dimensionals matrices and the number of
  // clumones in one is equal to the number of rows in two
  AS_LOG_S << "shapes for mat mult: ";
  if (log()) {
    stream_vector(one.shape());
  }
  AS_LOG_SA << " and ";
  if (log()) {
    stream_vector(two.shape());
  }
  AS_LOG_SA << std::endl;
  if (one.shape().size() != 2 || two.shape().size() != 2 ||
      one.shape()[1] != two.shape()[0]) {
    AS_LOG_S << "invalid shapes for mat mult " << std::endl;
    throw std::invalid_argument("shapes incompatible");
  }
  // "compute" resultshape
  Shape result_shape{one.shape()[0], two.shape()[1]};
  // create result layout
  std::shared_ptr<Layout> result_layout(
      createLayout(LAYOUT_TYPE::SIMPLE, result_shape));

  const auto& one_v = one.getValue();
  const std::vector<std::shared_ptr<U>>& two_v = two.getValue();
  size_t n_rows = two.shape()[0];
  size_t n_cols = two.shape()[1];
  // extract columns from two
  AS_LOG_S << "extracting columns; n_cols " << n_cols << " n_rows " << n_rows
           << std::endl;
  std::vector<std::vector<std::shared_ptr<U>>> cols;
  for (size_t i = 0; i < n_cols; ++i) {
    std::vector<std::shared_ptr<U>> col;
    for (size_t j = 0; j < n_rows; ++j) {
      AS_LOG_S << "i " << i << " j  " << j << std::endl;
      AS_LOG_S << i + j * n_cols << std::endl;
      col.push_back(two_v[i + j * n_cols]);
    }
    cols.push_back(col);
  }
  AS_LOG_S << "extracting columns done" << std::endl;

  // create the result vector
  std::vector<std::shared_ptr<HECtxt>> result_ctxts;
  result_ctxts.reserve(result_layout->size());

  // perform matrix multiplication
  // the entry i,j is the dot product of the ith row of one and the jth column
  // of two or in python/pseudocode: result[i,j] = dot(one[i:],two[:j])

  // number of columns in the lhs matrix
  size_t one_cols = one.shape()[1];
  AS_LOG_S << "starting dot prodcuts " << std::endl;
  for (size_t i = 0; i < result_shape[0]; ++i) {
    // columns are the inner loop so we can simply use pushback
    auto row_iter = std::make_pair<
        typename std::vector<std::shared_ptr<HECtxt>>::const_iterator,
        typename std::vector<std::shared_ptr<HECtxt>>::const_iterator>(
        one_v.begin() + i * one_cols, one_v.begin() + i * one_cols + one_cols);
    AS_LOG_S << "row " << i << " [" << i * n_cols << " : "
             << i * one_cols + one_cols << "]" << std::endl;
    for (size_t j = 0; j < result_shape[1]; ++j) {
      auto column_iter = std::make_pair<
          typename std::vector<std::shared_ptr<U>>::const_iterator,
          typename std::vector<std::shared_ptr<U>>::const_iterator>(
          cols[j].begin(), cols[j].end());
      AS_LOG_S << "col " << j << std::endl;
      result_ctxts.push_back(
          simple_dot_helper<
              typename std::vector<std::shared_ptr<HECtxt>>::const_iterator,
              typename std::vector<std::shared_ptr<U>>::const_iterator>(
              row_iter, column_iter)[0]);
    }
  }

  AS_LOG_S << "dot prodcuts done" << std::endl;
  std::stringstream result_name;
  result_name << one.getName() << " X " << two.getName();
  Ctxt result_ctxt = Ctxt(result_ctxts, result_layout, result_name.str());
  try {
    AS_LOG_S << "mat mult result: shape ";
    if (log()) {
      stream_vector(result_ctxt.shape());
    }
    AS_LOG_SA << std::endl;
    if (log()) {
      stream_vector(result_ctxt.decryptDouble());
    }
    AS_LOG_SA << std::endl;
  } catch (const std::exception& e) {
    AS_LOG_S << e.what() << std::endl;
  }
  return result_ctxt;
}

template <class T, class U>
Ctxt SimpleLayout::mat_mult_general_internal(const Ctxt& one,
                                             const T& two) const {
  AS_LOG_S << "mat_mutl_general not implemented yet" << std::endl;
  throw std::exception();
}

// others
Ptxt SimpleLayout::broadcast(const Ptxt& ptxt, const Shape& result_shape,
                             absl::Span<const int64_t> dimensions) const {
  AS_LOG_S << "broadcasting from shape: ";
  if (log()) {
    stream_vector(ptxt.shape());
  }
  AS_LOG_SA << " to shape ";
  if (log()) {
    stream_vector(result_shape);
  }
  AS_LOG_SA << "; dimensions : { ";
  for (const auto i : dimensions) {
    AS_LOG_SA << i << ", ";
  }
  AS_LOG_SA << " }" << std::endl;
  if (dimensions.size() != 1) {
    AS_LOG_S << "more than one broadcast dimension not supported at the moment";
    throw std::invalid_argument(
        "more than on broadcast dimension not supported at the moment");
  }
  // create reusult objecst
  std::shared_ptr<Layout> result_layout(
      createLayout(LAYOUT_TYPE::SIMPLE, result_shape));
  const std::vector<std::shared_ptr<HEPtxt>> ptxt_v = ptxt.getValue();
  std::vector<std::shared_ptr<HEPtxt>> result_ptxts(result_layout->size());
  // we'll iterate over this
  AS_LOG_S << "broadcasting " << std::endl;
  auto broadcast_dim = dimensions[0];
  // Shape idxs(result_shape.size(), 0);
  // for (size_t dim = 0; dim < result_shape.size(); ++dim) {
  //   if (dim == broadcast_dim) {
  //     for (size_t i = 0; i < result_shape[broadcast_dim]; ++i) {
  //       idxs[broadcast_dim] = i;
  //       AS_LOG_S << "";
  //       stream_vector(idxs);
  //       AS_LOG_SA << std::endl;
  //       size_t start = multi_index_to_flat(idxs, result_shape);
  //       size_t size = ptxt.layout().size();
  //       AS_LOG_S << "start: " << start << " size: " << size << std::endl;
  //       std::copy(ptxt_v.begin(), ptxt_v.begin() + size,
  //                 result_ptxts.begin() + start);
  //     }
  //   }
  // }

  // this is pretty much a copy of xla::Literl::Broadcast. just made some
  // modificitons that fit our purposes

  // first we need to make some conversion
  // TODO RP: converting the shapes back and forth between xla shapes and
  // aluminum_shark shapes should be addressed at some point
  xla::Shape xla_shape = create_xla_dummy_shape(ptxt.shape());
  xla::Shape xla_result_shape = create_xla_dummy_shape(result_shape);

  // scratch_source_index is temporary storage space for the computed index
  // into the input literal.  We put it here to avoid allocating an
  // std::vector in every iteration of ShapeUtil::ForEachIndex.
  std::vector<int64_t> scratch_source_index(xla_shape.dimensions_size());

  // do i need this?
  // for (int64_t i = 0; i < dimensions.size(); ++i) {
  //   int64_t dynamic_size = GetDynamicSize(i);
  //   result.SetDynamicSize(dimensions[i], dynamic_size);
  // }

  xla::ShapeUtil::ForEachIndex(
      xla_result_shape, [&](absl::Span<const int64_t> output_index) {
        for (int64_t i = 0, end = dimensions.size(); i < end; ++i) {
          scratch_source_index[i] = output_index[dimensions[i]];
        }
        int64_t dest_index = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            xla_result_shape, output_index);
        int64_t source_index =
            xla::IndexUtil::MultidimensionalIndexToLinearIndex(
                xla_shape, scratch_source_index);
        // memcpy(dest_data + primitive_size * dest_index,
        //        source_data + primitive_size * source_index, primitive_size);
        // this where the acutall broadcasting happens
        AS_LOG_S << "copying from " << source_index << " to " << dest_index
                 << std::endl;
        result_ptxts[dest_index] = ptxt_v[source_index];
        return true;
      });

  return Ptxt(result_ptxts, result_layout, ptxt.getName() + " broadcast");
}

// template instantiation
template Ctxt SimpleLayout::dot_internal<Ctxt, HECtxt>(const Ctxt& one,
                                                       const Ctxt& two) const;
template Ctxt SimpleLayout::mat_mult_internal<Ctxt, HECtxt>(
    const Ctxt& one, const Ctxt& two) const;
template Ctxt SimpleLayout::mat_mult_general_internal<Ctxt, HECtxt>(
    const Ctxt& one, const Ctxt& two) const;

template Ctxt SimpleLayout::dot_internal<Ptxt, HEPtxt>(const Ctxt& one,
                                                       const Ptxt& two) const;
template Ctxt SimpleLayout::mat_mult_internal<Ptxt, HEPtxt>(
    const Ctxt& one, const Ptxt& two) const;
template Ctxt SimpleLayout::mat_mult_general_internal<Ptxt, HEPtxt>(
    const Ctxt& one, const Ptxt& two) const;

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

// matrix and vector operations

// Dot product between two vectors
Ctxt BatchLayout::dot(const Ctxt& one, const Ctxt& two) const {
  AS_LOG_S << "not implemented yet" << std::endl;
}
Ctxt BatchLayout::dot(const Ctxt& one, const Ptxt& two) const {
  AS_LOG_S << "not implemented yet" << std::endl;
}

Ctxt BatchLayout::mat_mult(const Ctxt& one, const Ctxt& two) const {
  AS_LOG_S << "not implemented yet" << std::endl;
}
Ctxt BatchLayout::mat_mult(const Ctxt& one, const Ptxt& two) const {
  AS_LOG_S << "not implemented yet" << std::endl;
}
// More general matrix multplication for hihger dimensional matrices
// see: https://www.tensorflow.org/xla/operation_semantics#dotgeneral, and
// https://en.wikipedia.org/wiki/Tensor_contraction
Ctxt BatchLayout::mat_mult_general(const Ctxt& one, const Ctxt& two) const {
  AS_LOG_S << "not implemented yet" << std::endl;
}
Ctxt BatchLayout::mat_mult_general(const Ctxt& one, const Ptxt& two) const {
  AS_LOG_S << "not implemented yet" << std::endl;
}

// others
Ptxt BatchLayout::broadcast(const Ptxt& ptxt, const Shape& result_shape,
                            absl::Span<const int64_t> dimensions) const {
  AS_LOG_S << "not implemented yet" << std::endl;
  throw std::logic_error("not implemented yet");
}

// Free functions

Layout* createLayout(const char* type, const Shape& shape) {
  LAYOUT_TYPE lt = string_to_layout_type(type);
  return createLayout(lt, shape);
}

Layout* createLayout(const LAYOUT_TYPE type, const Shape& shape) {
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

// helpers
size_t multi_index_to_flat(const std::vector<size_t>& index,
                           const Shape& shape) {
  if (shape.size() != index.size()) {
    AS_LOG_S << "missmatching index: ";
    if (log()) {
      stream_vector(index);
    }
    AS_LOG_SA << " and shape: ";
    if (log()) {
      stream_vector(shape);
    }
    AS_LOG_SA << std::endl;
    throw std::invalid_argument("index and shape missmatch");
  }
  // const size_t = index.size();
  size_t ret = 1;
  size_t count = 0;
  for (auto i : index) {
    if (i >= shape[count++]) {
      AS_LOG_S << "index and shape missmatch" << std::endl;
      throw std::invalid_argument("index and shape missmatch");
    }
    ret *= i;
  }
  return ret;
}

xla::Shape create_xla_dummy_shape(const Shape& shape) {
  // the primitive type doesnt really matter so we just pick one
  std::vector<int64_t> cast_v(shape.begin(), shape.end());
  return xla::ShapeUtil::MakeShape(
      xla::PrimitiveType::F32,
      absl::Span<const int64_t>(cast_v.data(), cast_v.size()));
}

}  // namespace aluminum_shark