#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>

#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/he_backend/he_backend.h"
#include "tensorflow/compiler/plugin/aluminum_shark/layout.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/plugin/aluminum_shark/ptxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/utils/exception.h"
#include "tensorflow/compiler/plugin/aluminum_shark/utils/utils.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace aluminum_shark {

// things taken from hlo_evaluator_typed_visitor.h
// Creates a vector of multipliers which can be used to create a linear index
// into shape.
//
// Given the multidimensional index {i1, ..., iN} and
// M = MakeDimMultipliers(shape), the corresponding linear index LI is simply
//
//   LI = i1 * M[1] + i2 * M[2] + ... + iN * M[N].
//
// This lets you calculate LI given the multidimensional indices in any order.
static xla::DimensionVector MakeDimMultipliers(const xla::Shape& shape) {
  xla::DimensionVector v(shape.rank());
  int64_t scale = 1;
  for (auto dim : xla::LayoutUtil::MinorToMajor(shape)) {
    v[dim] = scale;
    scale *= shape.dimensions(dim);
  }
  return v;
}

// Batch Layout

void BatchLayout::init() {
  AS_LOG_S << "initialzing BatchLayout. shape:  ";
  if (log()) {
    stream_vector(shape_);
  }
  size_t bs = shape_[0];  // assumes batch dim is first
  // special case for vectors and scalars
  if (shape_.size() == 1) {
    bs = 1;
  }
  AS_LOG_SA << " size_: " << size_ << ", batch size: " << bs;
  size_t step_size = size_ / bs;
  AS_LOG_SA << ", step_size size: " << step_size << std::endl;

  axis_0_ = step_size;
  axis_1_ = bs;
  AS_LOG_S << "axis_0 " << axis_0_ << ", axis_1 " << axis_1_ << std::endl;
}

std::pair<size_t, size_t> BatchLayout::get_layout_index(size_t i) const {
  return std::pair<size_t, size_t>(i % axis_0_, i / axis_0_);
}

LAYOUT_TYPE BatchLayout::type() const { return LAYOUT_TYPE::BATCH; }

Layout* BatchLayout::deepCopy() const { return new BatchLayout(*this); }

// returns the actual shape of the underlying buffer
Shape BatchLayout::get_physical_shape() const {
  Shape temp = shape_;
  temp[0] = 1;
  return temp;
}

// Operation Interface
void BatchLayout::add_in_place(Ctxt& one, const Ctxt& two) const {
  auto& one_v = one.getValue();
  const auto& two_v = two.getValue();
  if (one_v.size() != two_v.size()) {
    AS_LOG_S << "incompatbile shapes: ";
    stream_vector(one.shape());
    AS_LOG_SA << " and ";
    stream_vector(two.shape());
    AS_LOG_SA << std::endl;
    throw std::runtime_error("incompatbile shapes");
  }
  // create function to run in prallel
  auto func = [&one_v, &two_v](size_t i) {
    one_v[i]->addInPlace(to_std_shared_ptr(two_v[i]));
  };
  run_parallel(0, one_v.size(), func);
  AS_LOG_INFO << "addition done " << std::endl;
}

void BatchLayout::multiply_in_place(Ctxt& one, const Ctxt& two) const {
  AS_LOG_INFO << "Ctxt *= Ctxt" << std::endl;
  auto& one_v = one.getValue();
  const auto& two_v = two.getValue();
  if (one_v.size() != two_v.size()) {
    AS_LOG_WARNING << "incompatbile shapes: ";
    stream_vector(one.shape());
    AS_LOG_SA << " and ";
    stream_vector(two.shape());
    AS_LOG_SA << std::endl;
    throw std::runtime_error("incompatbile shapes");
  }
  AS_LOG_INFO << "batch layout multiply in place, value sizes: " << one_v.size()
              << " and " << two_v.size() << std::endl;

  // create function to run in prallel
  auto func = [&one_v, &two_v](size_t i) {
    one_v[i]->multInPlace(to_std_shared_ptr(two_v[i]));
  };
  run_parallel(0, one_v.size(), func);
  AS_LOG_INFO << "multiplying done " << std::endl;
}

void BatchLayout::add_in_place(Ctxt& one, const Ptxt& two) const {
  AS_LOG_INFO << "Ctxt += Ptxt" << std::endl;
  Ptxt copy = two;
  copy.updateLayout(LAYOUT_TYPE::BATCH, one.getContext());
  const auto& two_v = copy.getValue();
  auto& one_v = one.getValue();

  if (one_v.size() != two_v.size()) {
    AS_LOG_CRITICAL << "incompatbile shapes: " << one.shape() << " and "
                    << two.shape() << ". lhs contains " << one_v.size()
                    << " values and rhs contains " << two_v.size() << "values"
                    << std::endl;
    throw std::runtime_error("incompatbile shapes");
  }
  // create function to run in prallel
  auto func = [&one_v, &two_v](size_t i) {
    one_v[i]->addInPlace(to_std_shared_ptr(two_v[i]));
  };
  AS_LOG_INFO << "running parallel inplace addition" << std::endl;
  run_parallel(0, one_v.size(), func);
}

void BatchLayout::multiply_in_place(Ctxt& one, const Ptxt& two) const {
  AS_LOG_INFO << "Ctxt *= Ptxt" << std::endl;
  Ptxt copy = two;
  copy.updateLayout(LAYOUT_TYPE::BATCH, one.getContext());
  const auto& two_v = copy.getValue();
  auto& one_v = one.getValue();
  if (one_v.size() != two_v.size()) {
    AS_LOG_CRITICAL << "incompatbile shapes: " << one.shape() << " and "
                    << two.shape() << std::endl;
    throw std::runtime_error("incompatbile shapes");
  }
  // create function to run in prallel
  auto func = [&one_v, &two_v](size_t i) {
    one_v[i]->multInPlace(to_std_shared_ptr(two_v[i]));
  };
  run_parallel(0, one_v.size(), func);
  AS_LOG_INFO << "multiplying done " << std::endl;
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

// matrix and vector operations

Ctxt BatchLayout::dot(const Ctxt& lhs, const Ctxt& rhs) const {
  // Dot
  return dot_internal<Ctxt, HECtxt>(lhs, rhs);
}

Ctxt BatchLayout::dot(const Ctxt& lhs, const Ptxt& rhs) const {
  Ptxt copy = rhs;
  copy.updateLayout(LAYOUT_TYPE::SIMPLE, lhs.getContext());
  const auto& two_v = copy.getValue();
  AS_LOG_INFO << "dot: lhs " << lhs.shape() << "rhs: " << copy.shape()
              << std::endl;
#ifdef LAYOUT_DEBUG
  AS_LOG_INFO << "dot: lhs \n"
              << PrintWithShape<double>(lhs.decryptDouble(), lhs.shape())
              << "rhs: \n"
              << PrintWithShape<double>(copy.decodeDouble(), copy.shape())
              << std::endl;
#endif
  return dot_internal<Ptxt, HEPtxt>(lhs, copy);
}

template <class T, class U>
Ctxt BatchLayout::dot_internal(const Ctxt& one, const T& two) const {
  // with the batchlayout and batch size b we treat the first dimension of
  // the lhs arguement as 1 so any matrix of the lhs of shape (b x m) is
  // treated as (1 x m). further a vector with n elements (shape (n)) is
  // acutally a stack of b vectors of shape (n). while on the lhs elements
  // are encoded in fewer ciphertexts the rhs we still need the full number
  // of elements

  // shape checks
  // a dot prodcut between (b X n) and (n) in this layout is actually just
  // (n) dot (n)
  if (one.shape().size() > 2 || two.shape().size() != 1) {
    // not a vector. run mat mult
    return mat_mult(one, two);
  }

  // check incompatible vector shapes
  if ((one.shape().size() == 2 &&
       one.shape()[1] !=
           two.shape()[0])  // if we are dealing with batched inputs we need
                            // to check the second dimensions of the lhs
                            // against the 1st of the rhs
      || one.shape()[0] != two.shape()[0]) {
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
  // "compute" resultshape. is either (b,1) or (b)
  Shape result_shape;
  if (one.shape().size() == 2) {
    result_shape = {one.shape()[0], 1};
  } else {
    result_shape = {one.shape()[0]};
  }

  // create result layout
  std::shared_ptr<Layout> result_layout(
      createLayout(LAYOUT_TYPE::BATCH, result_shape));
  const auto& one_v = one.getValue();
  const auto& two_v = two.getValue();

  // stick beginning and end iterators into a pair
  auto one_iters =
      std::make_pair<typename std::vector<shared_ptr<HECtxt>>::const_iterator,
                     typename std::vector<shared_ptr<HECtxt>>::const_iterator>(
          one_v.cbegin(), one_v.cend());
  auto two_iters =
      std::make_pair<typename std::vector<shared_ptr<U>>::const_iterator,
                     typename std::vector<shared_ptr<U>>::const_iterator>(
          two_v.cbegin(), two_v.cend());
  auto result_ctxts = simple_dot_helper<
      typename std::vector<shared_ptr<HECtxt>>::const_iterator,
      typename std::vector<shared_ptr<U>>::const_iterator>(one_iters,
                                                           two_iters);
  std::stringstream result_name;
  result_name << one.getName() << " dot " << two.getName();
  return Ctxt(result_ctxts, result_layout, result_name.str());
}

template <class T, class U>
Ctxt BatchLayout::mat_mult_internal(const Ctxt& one, const T& two) const {
  // shape checks
  // this only works for iif we have 2 dimensionals matrices and the number
  // of clumones in one is equal to the number of rows in two
  AS_LOG_INFO << "shapes for mat mult: " << one.shape() << ", " << two.shape()
              << std::endl;
  if (one.shape().size() != 2 || two.shape().size() != 2 ||
      one.shape()[1] != two.shape()[0]) {
    AS_LOG_S << "invalid shapes for mat mult " << std::endl;
    throw std::invalid_argument("shapes incompatible");
  }

  Shape result_shape{one.shape()[0], two.shape()[1]};
  AS_LOG_INFO << "result shape: " << result_shape << std::endl;

  // create function to compute the elements of the results
  const auto& lhs_v = one.getValue();
  const auto& rhs_v = two.getValue();
  AS_LOG_S << "lhs_v.size() = " << lhs_v.size()
           << ", rhs_v.size() = " << rhs_v.size() << std::endl;
  const auto& lhs_shape = one.shape();
  const auto& rhs_shape = two.shape();
  auto func = [&lhs_v, &rhs_v, &lhs_shape, &rhs_shape,
               &result_shape](const absl::Span<const int64_t> out_index) {
    AS_LOG_S << "output index: "
             << std::vector<int64_t>(out_index.begin(), out_index.end())
             << std::endl;
    // setup the varialbes we'll iterate over
    int row = out_index[0];
    int col = out_index[1];
    std::vector<size_t> lhs_index(2, 0);
    std::vector<size_t> rhs_index(2, 0);
    lhs_index[0] = row;
    rhs_index[1] = col;
    // iteration bound. we could either use lhs_shape[1] or rhs_shape[0]
    // here
    size_t n_iter = lhs_shape[1];

    shared_ptr<HECtxt> result;
    for (size_t i = 0; i < n_iter; ++i) {
      // iterate over the colmuns of lhs and rows of lhs
      lhs_index[1] = i;
      rhs_index[0] = i;
      AS_LOG_DEBUG << lhs_index << " * " << rhs_index << std::endl;
      AS_LOG_DEBUG << "lhs_v.size() = " << lhs_v.size()
                   << " index: " << multi_index_to_flat(lhs_index, lhs_shape)
                   << " , rhs_v.size() = " << rhs_v.size()
                   << " index: " << multi_index_to_flat(rhs_index, rhs_shape)
                   << std::endl;
      shared_ptr<HECtxt> temp =
          *(lhs_v[multi_index_to_flat(lhs_index, lhs_shape)]) *
          to_std_shared_ptr(rhs_v[multi_index_to_flat(rhs_index, rhs_shape)]);
#ifdef LAYOUT_DEBUG
      // decrypt values
      const HEContext* context = temp->getContext();
      AS_LOG_S << "decypting temp" << std::endl;
      std::vector<double> temp_dec = context->decryptDouble(temp);
      try {
        std::shared_ptr<HEPtxt> ptxt_p = std::dynamic_pointer_cast<HEPtxt>(
            rhs_v[multi_index_to_flat(rhs_index, rhs_shape)]);
        AS_LOG_S << "decypting lhs" << std::endl;
        std::vector<double> lhs_dec = context->decryptDouble(
            lhs_v[multi_index_to_flat(lhs_index, lhs_shape)]);
        AS_LOG_S << "decypting rhs" << std::endl;
        std::vector<double> rhs_dec = context->decodeDouble(ptxt_p);
        AS_LOG_S << "decrypted: \n "
                 << PrintWithShape<double>(
                        std::vector<double>(lhs_dec.begin(),
                                            lhs_dec.begin() + result_shape[0]),
                        {1, result_shape[0]})
                 << " * \n "
                 << PrintWithShape<double>(
                        std::vector<double>(rhs_dec.begin(),
                                            rhs_dec.begin() + result_shape[0]),
                        {1, result_shape[0]})
                 << " =  \n"
                 << PrintWithShape<double>(
                        std::vector<double>(temp_dec.begin(),
                                            temp_dec.begin() + result_shape[0]),
                        {1, result_shape[0]})
                 << std::endl;
      } catch (const std::exception& e) {
        AS_LOG_S << "something messed up" << std::endl;
        AS_LOG_S << e.what() << '\n';
      }

#endif
      if (!result) {  // first iteration
        AS_LOG_S << "First iteration" << std::endl;
        result = temp;
      } else {
#ifdef LAYOUT_DEBUG
        std::vector<double> result_dec = context->decryptDouble(result);
        AS_LOG_S << "decrypted: \n "
                 << PrintWithShape<double>(
                        std::vector<double>(
                            result_dec.begin(),
                            result_dec.begin() + result_shape[0]),
                        {1, result_shape[0]})
                 << " + \n "
                 << PrintWithShape<double>(
                        std::vector<double>(temp_dec.begin(),
                                            temp_dec.begin() + result_shape[0]),
                        {1, result_shape[0]})
                 << " = \n ";
#endif
        result->addInPlace(temp);
#ifdef LAYOUT_DEBUG
        result_dec = context->decryptDouble(result);
        AS_LOG_SA << PrintWithShape<double>(
                         std::vector<double>(
                             result_dec.begin(),
                             result_dec.begin() + result_shape[0]),
                         {1, result_shape[0]})
                  << std::endl;
#endif
      }
    }
    return result;
  };

  // create result objects
  // create result layout
  std::shared_ptr<Layout> result_layout(
      createLayout(LAYOUT_TYPE::BATCH, result_shape));
  // create the result vector
  std::vector<shared_ptr<HECtxt>> result_ctxts(two.shape()[1]);

  // create a "fake" for the output to iterate over. in this shape we'll
  // set the batch dimension to 1
  auto fake_shape = create_xla_dummy_shape({1, two.shape()[1]});

  // populate the ctxt vector
  std::vector<int64_t> base_vec(fake_shape.dimensions_size(), 0);
  std::vector<int64_t> incr_vec(fake_shape.dimensions_size(), 1);
#ifndef LAYOUT_DEBUG
  xla::ShapeUtil::ForEachIndexParallel(
#else
  // disable Parallel execution for debugging
  xla::ShapeUtil::ForEachIndex(
#endif
      fake_shape, /*base*/ base_vec, /*count*/ fake_shape.dimensions(),
      /*increment*/ incr_vec,
      [&result_ctxts, &fake_shape,
       &func](const absl::Span<const int64_t> multi_index) {
        auto linear_index = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            fake_shape, multi_index);
        result_ctxts[linear_index] = shared_ptr<HECtxt>(func(multi_index));
#ifdef LAYOUT_DEBUG
        return true;
#endif
      });

  AS_LOG_S << "dot prodcuts done" << std::endl;
  std::stringstream result_name;
  result_name << one.getName() << " X " << two.getName();
  Ctxt result_ctxt = Ctxt(result_ctxts, result_layout, result_name.str());
#ifdef LAYOUT_DEBUG
  try {
    AS_LOG_DEBUG << "mat mult result: " << ShapePrint(result_ctxt.shape())
                 << std::endl;
    AS_LOG_DEBUG << "decryptions: \n"
                 << PrintWithShape<double>(result_ctxt.decryptDouble(),
                                           result_ctxt.shape())
                 << std::endl;
  } catch (const decryption_error& e) {
    AS_LOG_S << e.what() << std::endl;
  } catch (const std::exception& e) {
    AS_LOG_S << e.what() << std::endl;
    throw;
  }
#endif
  return result_ctxt;
}

Ctxt BatchLayout::mat_mult(const Ctxt& one, const Ctxt& two) const {
  return mat_mult_internal<Ctxt, HECtxt>(one, two);
}
Ctxt BatchLayout::mat_mult(const Ctxt& one, const Ptxt& two) const {
  // creating a non const copy and update the layout
  Ptxt copy = two;
  copy.updateLayout(LAYOUT_TYPE::SIMPLE, one.getContext());
  return mat_mult_internal<Ptxt, HEPtxt>(one, copy);
}
// More general matrix multplication for hihger dimensional matrices
// see: https://www.tensorflow.org/xla/operation_semantics#dotgeneral, and
// https://en.wikipedia.org/wiki/Tensor_contraction
Ctxt BatchLayout::mat_mult_general(const Ctxt& one, const Ctxt& two) const {
  AS_LOG_S << "not implemented yet" << std::endl;
  throw std::logic_error("not implemented yet");
}
Ctxt BatchLayout::mat_mult_general(const Ctxt& one, const Ptxt& two) const {
  AS_LOG_S << "not implemented yet" << std::endl;
  throw std::logic_error("not implemented yet");
}

// others

Ctxt BatchLayout::convolution(const Ctxt& lhs, const Ptxt& rhs,
                              xla::HloInstruction* hlo) const {
  Ptxt rhs_copy = rhs;
  AS_LOG_INFO << "updating Layout " << std::endl;
  rhs_copy.updateLayout(LAYOUT_TYPE::SIMPLE, lhs.getContext());
  AS_LOG_INFO << "Layout updated. performing convolution " << std::endl;
  // this is an adapted copy of
  // xla::HloEvaluatorTypedVisitor::ConvolutionWithLiterals
  const auto& window = hlo->window();
  // we need to "fake out" the system by creating fake shapes where the
  // batch dimension is 1, for both the lhs and the result
  AS_LOG_INFO << "creating shapes " << std::endl;
  Shape temp = xla_shape_to_shark_shape(hlo->shape());
  temp[0] = 1;
  const xla::Shape& result_shape = create_xla_dummy_shape(temp);
  temp = xla_shape_to_shark_shape(hlo->operand(0)->shape());
  temp[0] = 1;
  const xla::Shape& lhs_shape = create_xla_dummy_shape(temp);
  const xla::Shape& rhs_shape = hlo->operand(1)->shape();

  TF_CHECK_OK(xla::ShapeUtil::ValidateShape(lhs_shape));
  TF_CHECK_OK(xla::ShapeUtil::ValidateShape(rhs_shape));

  const auto& dnums = hlo->convolution_dimension_numbers();
  const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
  CHECK_EQ(num_spatial_dims, dnums.input_spatial_dimensions_size());
  CHECK_EQ(num_spatial_dims, dnums.kernel_spatial_dimensions_size());
  CHECK_GE(num_spatial_dims, 0);
  CHECK_EQ(window.dimensions_size(), num_spatial_dims);

  std::vector<int64_t> window_dimension_sizes;
  for (auto i : dnums.kernel_spatial_dimensions()) {
    window_dimension_sizes.push_back(
        xla::ShapeUtil::GetDimension(rhs_shape, i));
  }

  const xla::Shape& window_shape = xla::ShapeUtil::MakeShape(
      rhs_shape.element_type(), window_dimension_sizes);

  xla::DimensionVector lhs_dim_multipliers = MakeDimMultipliers(lhs_shape);
  xla::DimensionVector rhs_dim_multipliers = MakeDimMultipliers(rhs_shape);

  auto& lhs_v = lhs.getValue();
  auto& rhs_v = rhs_copy.getValue();

  const int64_t feature_group_count = hlo->feature_group_count();
  const int64_t batch_group_count = hlo->batch_group_count();

  AS_LOG_INFO << "Convolution batch group count = " << batch_group_count
              << "\n\t"
              << "input_batch_dimension = " << dnums.input_batch_dimension()
              << "\n\t"
              << "output_batch_dimension = " << dnums.output_batch_dimension()
              << std::endl;
  AS_LOG_INFO << "creating computation function " << std::endl;
  auto func = [&window_shape, &dnums, &lhs_shape, &rhs_shape, &window,
               &lhs_dim_multipliers, &rhs_dim_multipliers, &lhs_v, &rhs_v,
               feature_group_count,
               batch_group_count](const absl::Span<const int64_t> out_index) {
    // Dimension number applicable for input (lhs).
    const int64_t input_batch_dim = dnums.input_batch_dimension();
    const int64_t input_z_dim = dnums.input_feature_dimension();
    // Dimension number applicable for kernel (rhs).
    const int64_t kernel_input_z_dim = dnums.kernel_input_feature_dimension();
    const int64_t kernel_output_z_dim = dnums.kernel_output_feature_dimension();
    // Dimension number applicable for output.
    const int64_t output_batch_dim = dnums.output_batch_dimension();
    const int64_t output_z_dim = dnums.output_feature_dimension();

    const int64_t input_z_size =
        xla::ShapeUtil::GetDimension(lhs_shape, input_z_dim);

    const int64_t input_batch_size =
        xla::ShapeUtil::GetDimension(lhs_shape, input_batch_dim);

    const int64_t batch_group_size = input_batch_size / batch_group_count;

    // The size of an input feature group.
    const int64_t input_feature_group_size = input_z_size / feature_group_count;

    const int64_t output_z_size =
        xla::ShapeUtil::GetDimension(rhs_shape, kernel_output_z_dim);
    // The output feature dimension is a concatenation of convolution
    // results from the different groups.
    const int64_t output_feature_group_size =
        output_z_size / feature_group_count;

    // Calculate the group index to which the current output index
    // belongs.
    const int64_t feature_group_index =
        out_index[output_z_dim] / output_feature_group_size;

    const int64_t depthwise_multiplier =
        batch_group_count > 1 ? output_z_size / input_batch_size : 1;
    const int64_t batch_group_index =
        out_index[output_z_dim] / depthwise_multiplier;

    xla::DimensionVector rhs_spatial_index(
        dnums.kernel_spatial_dimensions_size(), 0);

    bool first = true;
    shared_ptr<HECtxt> result;

    // Convolve input feature with kernel.
    // The mechanism indexes into the correct LHS (input) and RHS (kernel)
    // locations and accumulates multiplications for a given output index.
    do {
      // Find corresponding spatial dimension index for input (lhs).
      int64_t lhs_linear_spatial_index = 0;
      int64_t rhs_linear_spatial_index = 0;
      for (int64_t ki = 0; ki < rhs_spatial_index.size(); ++ki) {
        // Spatial dimension number for input (lhs) and output.
        const int64_t input_spatial_dim = dnums.input_spatial_dimensions(ki);
        const int64_t output_spatial_dim = dnums.output_spatial_dimensions(ki);

        // Calculate lhs (input) index without taking base dilation into
        // account.
        const auto& window_dim = window.dimensions(ki);
        const int64_t undilated_index =
            out_index[output_spatial_dim] * window_dim.stride() -
            window_dim.padding_low() +
            rhs_spatial_index[ki] * window_dim.window_dilation();
        // Skip if the lhs (input) index is to be dilated.  As an
        // optimization, skip this mod if there's no dilation.
        if (window_dim.base_dilation() > 1 &&
            undilated_index % window_dim.base_dilation() != 0) {
          goto cnt;
        }

        // Calculate the actual lhs (input) index after dilation.  As an
        // optimization, skip this integer divide if there's no dilation.
        int64_t lhs_spatial_index;
        if (window_dim.base_dilation() > 1) {
          lhs_spatial_index = undilated_index / window_dim.base_dilation();
        } else {
          lhs_spatial_index = undilated_index;
        }

        // Skip if input index is not in bounds.
        if (!(lhs_spatial_index >= 0 &&
              lhs_spatial_index < lhs_shape.dimensions(input_spatial_dim))) {
          goto cnt;
        }

        lhs_linear_spatial_index +=
            lhs_spatial_index * lhs_dim_multipliers[input_spatial_dim];
        rhs_linear_spatial_index +=
            (window_dim.window_reversal()
                 ? ((window_dim.size() - 1) - rhs_spatial_index[ki])
                 : rhs_spatial_index[ki]) *
            rhs_dim_multipliers[dnums.kernel_spatial_dimensions(ki)];
      }

      for (int64_t rhs_iz = 0; rhs_iz < input_feature_group_size; ++rhs_iz) {
        const int64_t iz =
            feature_group_index * input_feature_group_size + rhs_iz;

        int64_t lhs_linear_index = lhs_linear_spatial_index;
        lhs_linear_index +=
            out_index[output_batch_dim] * lhs_dim_multipliers[input_batch_dim];

        // We are scraping only the diagonal elements in the resultant
        // convolution output when batch_group_count is greater than 1,
        // where 1 is the default. No scraping is done in that case.
        // This approach works out automatically for 'groups' in batches
        // with group_size > 1, because we already descend down the batch
        // dimension for the 'output_batch_dim' above.
        lhs_linear_index +=
            ((batch_group_index * batch_group_size) % input_batch_size) *
            lhs_dim_multipliers[input_batch_dim];

        lhs_linear_index += iz * lhs_dim_multipliers[input_z_dim];
        int64_t rhs_linear_index = rhs_linear_spatial_index;

        rhs_linear_index +=
            out_index[output_z_dim] * rhs_dim_multipliers[kernel_output_z_dim];
        rhs_linear_index += rhs_iz * rhs_dim_multipliers[kernel_input_z_dim];
        shared_ptr<HECtxt> temp =
            lhs_v[lhs_linear_index]->operator*(rhs_v[rhs_linear_index]);
        if (first) {
          result = temp;
          first = false;
        } else {
          result->addInPlace(temp);
        }
      }
    cnt : {}
    } while (xla::IndexUtil::BumpIndices(window_shape,
                                         absl::MakeSpan(rhs_spatial_index)));
    return result;
  };

  // create the result object
  Layout* layout =
      createLayout(LAYOUT_TYPE::BATCH, xla_shape_to_shark_shape(hlo->shape()));

  std::vector<shared_ptr<HECtxt>> ctxt_vector(layout->size() /
                                              layout->shape()[0]);
  // populate the ctxt vector
  AS_LOG_S << "created result vector with " << ctxt_vector.size() << " elements"
           << std::endl;
  std::vector<int64_t> base_vec(result_shape.dimensions_size(), 0);
  std::vector<int64_t> incr_vec(result_shape.dimensions_size(), 1);
  AS_LOG_S << "running convolution from " << base_vec << " to "
           << std::vector<int64_t>(result_shape.dimensions().begin(),
                                   result_shape.dimensions().end())
           << " incrementing by " << incr_vec << std::endl;
  xla::ShapeUtil::ForEachIndexParallel(
      result_shape, /*base*/ base_vec, /*count*/ result_shape.dimensions(),
      /*increment*/ incr_vec,
      [&ctxt_vector, &result_shape,
       &func](const absl::Span<const int64_t> multi_index) {
        auto linear_index = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            result_shape, multi_index);
        if (linear_index % 1000 == 0) {
          AS_LOG_INFO << "running "
                      << std::vector<int64_t>(multi_index.begin(),
                                              multi_index.end())
                      << " / "
                      << std::vector<int64_t>(result_shape.dimensions().begin(),
                                              result_shape.dimensions().end())
                      << std::endl;
        }
        ctxt_vector[linear_index] = func(multi_index);
      });

  return Ctxt(ctxt_vector, std::shared_ptr<Layout>(layout),
              "conv(" + lhs.getName() + ")");
}

// memory optimization stuff

// simple typedef for the execution plan data structure.
// the index `i` into the outermost vector is the index into the priority side.
// the pairs hold the output index `o` in pair.first and the minority index `j`
// in pair.second.
//
// Given an output vector `Y` and `rhs` and `lhs` as the inputs lhs prioiryt is
// computed as (simplified):
//
//          Y[o] += lhs[i] * rhs[j]
//
// and rhs priority is:
//
//          Y[o] += lhs[j] * rhs[i]

using ExecutionPlan = std::vector<std::vector<std::pair<int64_t, int64_t>>>;

Ctxt BatchLayout::mat_mult_memoptimized(Ctxt& one, Ptxt& two) const {
  // we first do a "dry run"  create pairs of indices that that tell us which
  // plaintexts are involoved in which result value. from these pairs we can
  // create create an execution order that allows us to release plaintexts as
  // early as possible

  // shape checks
  // this only works for iif we have 2 dimensionals matrices and the number
  // of clumones in one is equal to the number of rows in two
  AS_LOG_INFO << "shapes for mat mult: " << one.shape() << ", " << two.shape()
              << std::endl;
  if (one.shape().size() != 2 || two.shape().size() != 2 ||
      one.shape()[1] != two.shape()[0]) {
    AS_LOG_S << "invalid shapes for mat mult " << std::endl;
    throw std::invalid_argument("shapes incompatible");
  }

  two.updateLayout(LAYOUT_TYPE::SIMPLE, one.getContext());
  // the logical shape of the output after decoding is done
  Shape logical_shape{one.shape()[0], two.shape()[1]};
  AS_LOG_INFO << "result shape: " << logical_shape << std::endl;
  // this the actual phyical shape of the result
  auto result_shape = create_xla_dummy_shape({1, two.shape()[1]});

  // create function to do the dry and generate our pairs
  int64_t size = two.shape()[1];  // number of output ctxts. batch dim is 1

  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
      dot_indices(
          size,
          std::make_pair<std::vector<int64_t>, std::vector<int64_t>>({}, {}));

  const auto& lhs_shape = one.shape();
  const auto& rhs_shape = two.shape();
  auto func = [&dot_indices, &lhs_shape, &rhs_shape,
               &result_shape](const absl::Span<const int64_t> out_index) {
    AS_LOG_DEBUG << "output index: "
                 << std::vector<int64_t>(out_index.begin(), out_index.end())
                 << std::endl;
    // setup the varialbes we'll iterate over
    int row = out_index[0];
    int col = out_index[1];
    std::vector<size_t> lhs_index(2, 0);
    std::vector<size_t> rhs_index(2, 0);
    lhs_index[0] = row;
    rhs_index[1] = col;
    // iteration bound. we could either use lhs_shape[1] or rhs_shape[0]
    // here
    size_t n_iter = lhs_shape[1];

    for (size_t i = 0; i < n_iter; ++i) {
      // iterate over the colmuns of lhs and rows of lhs
      lhs_index[1] = i;
      rhs_index[0] = i;
      auto flat_outindex = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
          result_shape, out_index);
      // this is where we record the indecies
      dot_indices[flat_outindex].first.push_back(
          multi_index_to_flat(lhs_index, lhs_shape));
      dot_indices[flat_outindex].second.push_back(
          multi_index_to_flat(rhs_index, rhs_shape));
    }
  };

  // start the dry and extract the index pairs
  std::vector<int64_t> base_vec(result_shape.dimensions_size(), 0);
  std::vector<int64_t> incr_vec(result_shape.dimensions_size(), 1);
  xla::ShapeUtil::ForEachIndexParallel(
      result_shape, /*base*/ base_vec, /*count*/ result_shape.dimensions(),
      /*increment*/ incr_vec,
      [&func](const absl::Span<const int64_t> multi_index) {
        func(multi_index);
      });

  if (log(AS_DEBUG)) {
    std::stringstream ss;
    ss << "indicies generated: " << std::endl;
    for (auto& pair : dot_indices) {
      ss << "\t" << pair.first << std::endl;
      ss << "\t" << pair.second << std::endl << std::endl;
    }
    AS_LOG_DEBUG << "generated dot indicies. " << ss.str() << std::endl;
  }
  AS_LOG_INFO << "generated dot indicies" << std::endl;

  // input values
  auto& lhs_v = one.getValue();
  auto& rhs_v = two.getValue();
  AS_LOG_INFO << "lhs_v.size() = " << lhs_v.size()
              << ", rhs_v.size() = " << rhs_v.size() << std::endl;

  // create result objects
  // create result layout
  std::shared_ptr<Layout> result_layout(
      createLayout(LAYOUT_TYPE::BATCH, {one.shape()[0], two.shape()[1]}));
  // create the result vector
  std::vector<shared_ptr<HECtxt>> result_vector(two.shape()[1]);

  // group by rhs index but keep output index information
  // this needs a good amount of locking

  // the index into this vector corresponds to the index into the rhs data
  // each index holds a vector of std::pair<int64_t, int64_t>
  // pair.first is the output_index and pair.second is the index into the lhs
  // data. read and write access protect by `rhs_mutex`
  std::mutex rhs_mutex;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> rhs_indicies(
      rhs_v.size());

  size_t result_index = 0;
  for (auto& pair : dot_indices) {
    for (size_t i = 0; i < pair.second.size(); ++i) {
      int64_t lhs_index = pair.first[i];
      int64_t rhs_index = pair.second[i];
      rhs_indicies[rhs_index].push_back(
          std::pair<int64_t, int64_t>(result_index, lhs_index));
    }
    ++result_index;
  }
  AS_LOG_INFO << "generated rhs first index pairs" << std::endl;

  // the result vector holds a bunch of shared_ptr<HECtxt> objects.
  // The objects that go into it need to summed up. read and write
  // access is protected by the `result_mutex`. If the pointer at any
  // index `i` is not null it is responsiblity of the thread that
  // wants to write to `i` sum the object there and the object it
  // wants to write. to do so it must aquire the lock. take the
  // object out of the list and release the lock. the sumation must
  // not be performed while holding the lock. a theard can only write
  // to `i` iff the value at `i` == nullptr and it holds
  // `result_mutex`

  // create the result object
  std::mutex result_mutex;

  // create function to perform the actual dot;
  auto dot_func = [&rhs_mutex, &result_mutex, &rhs_indicies, &result_vector,
                   &lhs_v, &rhs_v]() -> bool {
    // get the index pair and the shared_ptr
    std::pair<int64_t, int64_t> rhs_pair;
    shared_ptr<HEPtxt> rhs_ptxt;

    int64_t rhs_index = 0;
    std::stringstream ss;
    AS_LOG_DEBUG << "entering first critical section" << std::endl;
    {  // start rhs_mutex
      std::lock_guard<std::mutex> lock(rhs_mutex);
      // find the first non_empty index
      for (; rhs_index < rhs_indicies.size(); ++rhs_index) {
        if (!rhs_indicies[rhs_index].empty()) {
          // take the last element and remove it
          rhs_pair = std::move(rhs_indicies[rhs_index].back());
          rhs_indicies[rhs_index].pop_back();
          break;
        }
      }
      // check if we consumed all items
      if (rhs_index == rhs_indicies.size()) {
        AS_LOG_DEBUG
            << "exiting first critical section and returning. nothing more todo"
            << std::endl;
        return false;
      }
      rhs_ptxt = rhs_v[rhs_index];
      // check if we took the last element and if so set the pointer to null
      if (rhs_indicies[rhs_index].empty()) {
        rhs_v[rhs_index] = nullptr;
      }
    }  // end rhs_mutex
    AS_LOG_DEBUG << "exiting first critical section: ptxt[ " << rhs_index
                 << "] " << rhs_ptxt << std::endl;

    // get lhs data  and output index
    int64_t output_index = rhs_pair.first;
    int64_t lhs_index = rhs_pair.second;
    output_index = rhs_pair.first;
    lhs_index = rhs_pair.second;

    ss << "getting lhs ciphertext" << lhs_index << "/" << lhs_v.size()
       << std::endl;
    AS_LOG_DEBUG << ss.str();
    shared_ptr<HECtxt> lhs_ctxt = lhs_v[lhs_index];
    ss << "got ctxt " << lhs_ctxt << " at index " << lhs_index << std::endl;
    AS_LOG_DEBUG << ss.str();

    // run ctxt plaintext multiplication
    AS_LOG_DEBUG << "running multiplication" << std::endl;
    shared_ptr<HECtxt> result = lhs_ctxt->operator*(rhs_ptxt);
    ss << "multiplication result ctxt " << result << std::endl;
    AS_LOG_DEBUG << ss.str();

    // write out result
    // we loop here because other threads might have written a result while we
    // were busy adding prior results to our result. if that is the case we
    // need to check again if we can write to the vector
    while (true) {
      // first aquire the lock
      shared_ptr<HECtxt> prior_result;  // to hold values that are allready in
                                        // the result vector;
      AS_LOG_DEBUG << "entering second critical section" << std::endl;
      {  // start result_mutex
        std::lock_guard<std::mutex> lock(result_mutex);
        // check if we can just write to result vector
        if (!result_vector[output_index]) {
          result_vector[output_index].swap(result);
          // we are done here
          AS_LOG_DEBUG << "exiting second critical section and returing"
                       << std::endl;
          return true;
        }
        // get the pointer of the resulut vector
        prior_result.swap(result_vector[output_index]);
        // result_vector[output_index] now holds a nullptr
      }  // end result_mutex
      AS_LOG_DEBUG << "exiting second critical section" << std::endl;
      // perform addition
      result->addInPlace(prior_result);
    }
  };

  // populate the ctxt vector
  run_parallel(dot_func);

  AS_LOG_INFO << "dot prodcuts done" << std::endl;
  std::stringstream result_name;
  result_name << one.getName() << " X " << two.getName();
  return Ctxt(result_vector, result_layout, result_name.str());
}

Ctxt BatchLayout::convolution_memoptimized(Ctxt& lhs, Ptxt& rhs,
                                           xla::HloInstruction* hlo) const {
  AS_LOG_INFO << "updating Layout " << std::endl;
  rhs.updateLayout(LAYOUT_TYPE::SIMPLE, lhs.getContext());
  AS_LOG_INFO << "Layout updated. performing convolution " << std::endl;
  // this is an adapted copy of
  // xla::HloEvaluatorTypedVisitor::ConvolutionWithLiterals
  const auto& window = hlo->window();
  // we need to "fake out" the system by creating fake shapes where the
  // batch dimension is 1, for both the lhs and the result
  AS_LOG_INFO << "creating shapes " << std::endl;
  Shape temp = xla_shape_to_shark_shape(hlo->shape());
  temp[0] = 1;
  const xla::Shape& result_shape = create_xla_dummy_shape(temp);
  temp = xla_shape_to_shark_shape(hlo->operand(0)->shape());
  temp[0] = 1;
  const xla::Shape& lhs_shape = create_xla_dummy_shape(temp);
  const xla::Shape& rhs_shape = hlo->operand(1)->shape();

  TF_CHECK_OK(xla::ShapeUtil::ValidateShape(lhs_shape));
  TF_CHECK_OK(xla::ShapeUtil::ValidateShape(rhs_shape));

  const auto& dnums = hlo->convolution_dimension_numbers();
  const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
  CHECK_EQ(num_spatial_dims, dnums.input_spatial_dimensions_size());
  CHECK_EQ(num_spatial_dims, dnums.kernel_spatial_dimensions_size());
  CHECK_GE(num_spatial_dims, 0);
  CHECK_EQ(window.dimensions_size(), num_spatial_dims);

  std::vector<int64_t> window_dimension_sizes;
  for (auto i : dnums.kernel_spatial_dimensions()) {
    window_dimension_sizes.push_back(
        xla::ShapeUtil::GetDimension(rhs_shape, i));
  }

  const xla::Shape& window_shape = xla::ShapeUtil::MakeShape(
      rhs_shape.element_type(), window_dimension_sizes);

  xla::DimensionVector lhs_dim_multipliers = MakeDimMultipliers(lhs_shape);
  xla::DimensionVector rhs_dim_multipliers = MakeDimMultipliers(rhs_shape);

  auto& lhs_v = lhs.getValue();
  auto& rhs_v = rhs.getValue();

  const int64_t feature_group_count = hlo->feature_group_count();
  const int64_t batch_group_count = hlo->batch_group_count();

  AS_LOG_INFO << "Convolution batch group count = " << batch_group_count
              << "\n\t"
              << "input_batch_dimension = " << dnums.input_batch_dimension()
              << "\n\t"
              << "output_batch_dimension = " << dnums.output_batch_dimension()
              << std::endl;
  AS_LOG_INFO << "creating computation function " << std::endl;

  // generate a vector that holds the lhs and rhs indecies. the vector holds a
  // pair of vector. the vectors on each side of the pair need to be multiplied
  // element wise and summed
  int64_t size = 1;
  for (size_t i = 1; i < result_shape.dimensions().size(); i++) {
    size *= result_shape.dimensions()[i];
  }

  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
      conv_indices(
          size,
          std::make_pair<std::vector<int64_t>, std::vector<int64_t>>({}, {}));

  auto func = [&window_shape, &dnums, &lhs_shape, &rhs_shape, &window,
               &lhs_dim_multipliers, &rhs_dim_multipliers, &conv_indices,
               feature_group_count, batch_group_count,
               &result_shape](const absl::Span<const int64_t> out_index) {
    // Dimension number applicable for input (lhs).
    const int64_t input_batch_dim = dnums.input_batch_dimension();
    const int64_t input_z_dim = dnums.input_feature_dimension();
    // Dimension number applicable for kernel (rhs).
    const int64_t kernel_input_z_dim = dnums.kernel_input_feature_dimension();
    const int64_t kernel_output_z_dim = dnums.kernel_output_feature_dimension();
    // Dimension number applicable for output.
    const int64_t output_batch_dim = dnums.output_batch_dimension();
    const int64_t output_z_dim = dnums.output_feature_dimension();

    const int64_t input_z_size =
        xla::ShapeUtil::GetDimension(lhs_shape, input_z_dim);

    const int64_t input_batch_size =
        xla::ShapeUtil::GetDimension(lhs_shape, input_batch_dim);

    const int64_t batch_group_size = input_batch_size / batch_group_count;

    // The size of an input feature group.
    const int64_t input_feature_group_size = input_z_size / feature_group_count;

    const int64_t output_z_size =
        xla::ShapeUtil::GetDimension(rhs_shape, kernel_output_z_dim);
    // The output feature dimension is a concatenation of convolution
    // results from the different groups.
    const int64_t output_feature_group_size =
        output_z_size / feature_group_count;

    // Calculate the group index to which the current output index
    // belongs.
    const int64_t feature_group_index =
        out_index[output_z_dim] / output_feature_group_size;

    const int64_t depthwise_multiplier =
        batch_group_count > 1 ? output_z_size / input_batch_size : 1;
    const int64_t batch_group_index =
        out_index[output_z_dim] / depthwise_multiplier;

    xla::DimensionVector rhs_spatial_index(
        dnums.kernel_spatial_dimensions_size(), 0);

    // Convolve input feature with kernel.
    // The mechanism indexes into the correct LHS (input) and RHS
    // (kernel) locations and accumulates multiplications for a given
    // output index.
    do {
      // Find corresponding spatial dimension index for input (lhs).
      int64_t lhs_linear_spatial_index = 0;
      int64_t rhs_linear_spatial_index = 0;
      for (int64_t ki = 0; ki < rhs_spatial_index.size(); ++ki) {
        // Spatial dimension number for input (lhs) and output.
        const int64_t input_spatial_dim = dnums.input_spatial_dimensions(ki);
        const int64_t output_spatial_dim = dnums.output_spatial_dimensions(ki);

        // Calculate lhs (input) index without taking base dilation into
        // account.
        const auto& window_dim = window.dimensions(ki);
        const int64_t undilated_index =
            out_index[output_spatial_dim] * window_dim.stride() -
            window_dim.padding_low() +
            rhs_spatial_index[ki] * window_dim.window_dilation();
        // Skip if the lhs (input) index is to be dilated.  As an
        // optimization, skip this mod if there's no dilation.
        if (window_dim.base_dilation() > 1 &&
            undilated_index % window_dim.base_dilation() != 0) {
          goto cnt;
        }

        // Calculate the actual lhs (input) index after dilation.  As an
        // optimization, skip this integer divide if there's no
        // dilation.
        int64_t lhs_spatial_index;
        if (window_dim.base_dilation() > 1) {
          lhs_spatial_index = undilated_index / window_dim.base_dilation();
        } else {
          lhs_spatial_index = undilated_index;
        }

        // Skip if input index is not in bounds.
        if (!(lhs_spatial_index >= 0 &&
              lhs_spatial_index < lhs_shape.dimensions(input_spatial_dim))) {
          goto cnt;
        }

        lhs_linear_spatial_index +=
            lhs_spatial_index * lhs_dim_multipliers[input_spatial_dim];
        rhs_linear_spatial_index +=
            (window_dim.window_reversal()
                 ? ((window_dim.size() - 1) - rhs_spatial_index[ki])
                 : rhs_spatial_index[ki]) *
            rhs_dim_multipliers[dnums.kernel_spatial_dimensions(ki)];
      }

      for (int64_t rhs_iz = 0; rhs_iz < input_feature_group_size; ++rhs_iz) {
        const int64_t iz =
            feature_group_index * input_feature_group_size + rhs_iz;

        int64_t lhs_linear_index = lhs_linear_spatial_index;
        lhs_linear_index +=
            out_index[output_batch_dim] * lhs_dim_multipliers[input_batch_dim];

        // We are scraping only the diagonal elements in the resultant
        // convolution output when batch_group_count is greater than 1,
        // where 1 is the default. No scraping is done in that case.
        // This approach works out automatically for 'groups' in batches
        // with group_size > 1, because we already descend down the
        // batch dimension for the 'output_batch_dim' above.
        lhs_linear_index +=
            ((batch_group_index * batch_group_size) % input_batch_size) *
            lhs_dim_multipliers[input_batch_dim];

        lhs_linear_index += iz * lhs_dim_multipliers[input_z_dim];
        int64_t rhs_linear_index = rhs_linear_spatial_index;

        rhs_linear_index +=
            out_index[output_z_dim] * rhs_dim_multipliers[kernel_output_z_dim];
        rhs_linear_index += rhs_iz * rhs_dim_multipliers[kernel_input_z_dim];

        auto flat_outindex = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            result_shape, out_index);

        // this is where we record the indecies
        conv_indices[flat_outindex].first.push_back(lhs_linear_index);
        conv_indices[flat_outindex].second.push_back(rhs_linear_index);
      }
    cnt : {}
    } while (xla::IndexUtil::BumpIndices(window_shape,
                                         absl::MakeSpan(rhs_spatial_index)));
    // return result;
  };

  std::vector<int64_t> base_vec(result_shape.dimensions_size(), 0);
  std::vector<int64_t> incr_vec(result_shape.dimensions_size(), 1);

  xla::ShapeUtil::ForEachIndexParallel(
      result_shape, /*base*/ base_vec, /*count*/ result_shape.dimensions(),
      /*increment*/ incr_vec,
      [&result_shape, &func](const absl::Span<const int64_t> multi_index) {
        auto linear_index = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            result_shape, multi_index);
        if (linear_index % 1000 == 0) {
          AS_LOG_INFO << "generating indecies "
                      << std::vector<int64_t>(multi_index.begin(),
                                              multi_index.end())
                      << " / "
                      << std::vector<int64_t>(result_shape.dimensions().begin(),
                                              result_shape.dimensions().end())
                      << std::endl;
        }
        func(multi_index);
      });

  if (log(AS_DEBUG)) {
    std::stringstream ss;
    ss << "indicies generated: " << std::endl;
    for (auto& pair : conv_indices) {
      ss << "\t" << pair.first << std::endl;
      ss << "\t" << pair.second << std::endl << std::endl;
    }
    AS_LOG_DEBUG << "generated convolution indicies. " << ss.str() << std::endl;
  }
  AS_LOG_INFO << "generated convolution indicies" << std::endl;
  // create the result object
  Layout* layout =
      createLayout(LAYOUT_TYPE::BATCH, xla_shape_to_shark_shape(hlo->shape()));
  std::vector<shared_ptr<HECtxt>> result_vector(layout->size() /
                                                layout->shape()[0]);

  // group by rhs index but keep output index information
  // this needs a good amount of locking

  // the index into this vector corresponds to the index into the rhs data
  // each index holds a vector of std::pair<int64_t, int64_t>
  // pair.first is the output_index and pair.second is the index into the lhs
  // data. read and write access protect by `rhs_mutex`
  std::mutex rhs_mutex;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> rhs_indicies(
      rhs_v.size());

  size_t result_index = 0;
  for (auto& pair : conv_indices) {
    for (size_t i = 0; i < pair.second.size(); ++i) {
      int64_t lhs_index = pair.first[i];
      int64_t rhs_index = pair.second[i];
      rhs_indicies[rhs_index].push_back(
          std::pair<int64_t, int64_t>(result_index, lhs_index));
    }
    ++result_index;
  }
  AS_LOG_INFO << "generated rhs first index pairs" << std::endl;

  if (log(AS_DEBUG)) {
    std::stringstream ss;
    ss << "pairs generated: " << std::endl;
    size_t i = 0;
    for (auto& vec : rhs_indicies) {
      ss << "\t" << i++ << ": ";
      for (auto& pair : vec) {
        ss << "<" << pair.first << ", " << pair.second << "> ";
      }
      ss << std::endl;
    }
    AS_LOG_DEBUG << ss.str() << std::endl;
  }

  // the result vector holds a bunch of shared_ptr<HECtxt> objects.
  // The objects that go into it need to summed up. read and write
  // access is protected by the `result_mutex`. If the pointer at any
  // index `i` is not null it is responsiblity of the thread that
  // wants to write to `i` sum the object there and the object it
  // wants to write. to do so it must aquire the lock. take the
  // object out of the list and release the lock. the sumation must
  // not be performed while holding the lock. a theard can only write
  // to `i` iff the value at `i` == nullptr and it holds
  // `result_mutex`

  // create the result object
  std::mutex result_mutex;

  // create function to perform the actual convolution;
  auto conv_func = [&rhs_mutex, &result_mutex, &rhs_indicies, &result_vector,
                    &lhs_v, &rhs_v]() -> bool {
    // get the index pair and the shared_ptr
    std::pair<int64_t, int64_t> rhs_pair;
    shared_ptr<HEPtxt> rhs_ptxt;

    int64_t rhs_index = 0;
    std::stringstream ss;
    AS_LOG_DEBUG << "entering first critical section" << std::endl;
    {  // start rhs_mutex
      std::lock_guard<std::mutex> lock(rhs_mutex);
      // find the first non_empty index
      for (; rhs_index < rhs_indicies.size(); ++rhs_index) {
        if (!rhs_indicies[rhs_index].empty()) {
          // take the last element and remove it
          rhs_pair = std::move(rhs_indicies[rhs_index].back());
          rhs_indicies[rhs_index].pop_back();
          break;
        }
      }
      // check if we consumed all items
      if (rhs_index == rhs_indicies.size()) {
        AS_LOG_DEBUG
            << "exiting first critical section and returning. nothing more todo"
            << std::endl;
        return false;
      }
      rhs_ptxt = rhs_v[rhs_index];
      // check if we took the last element and if so set the pointer to null
      if (rhs_indicies[rhs_index].empty()) {
        rhs_v[rhs_index] = nullptr;
      }
    }  // end rhs_mutex
    AS_LOG_DEBUG << "exiting first critical section: ptxt[ " << rhs_index
                 << "] " << rhs_ptxt << std::endl;

    // get lhs data  and output index
    int64_t output_index = rhs_pair.first;
    int64_t lhs_index = rhs_pair.second;
    output_index = rhs_pair.first;
    lhs_index = rhs_pair.second;

    ss << "getting lhs ciphertext" << lhs_index << "/" << lhs_v.size()
       << std::endl;
    AS_LOG_DEBUG << ss.str();
    shared_ptr<HECtxt> lhs_ctxt = lhs_v[lhs_index];
    ss << "got ctxt " << lhs_ctxt << " at index " << lhs_index << std::endl;
    AS_LOG_DEBUG << ss.str();

    // run ctxt plaintext multiplication
    AS_LOG_DEBUG << "running multiplication" << std::endl;
    shared_ptr<HECtxt> result = lhs_ctxt->operator*(rhs_ptxt);
    ss << "multiplication result ctxt " << result << std::endl;
    AS_LOG_DEBUG << ss.str();

    // write out result
    // we loop here because other threads might have written a result while we
    // were busy adding prior results to our result. if that is the case we
    // need to check again if we can write to the vector
    while (true) {
      // first aquire the lock
      shared_ptr<HECtxt> prior_result;  // to hold values that are allready in
                                        // the result vector;
      AS_LOG_DEBUG << "entering second critical section" << std::endl;
      {  // start result_mutex
        std::lock_guard<std::mutex> lock(result_mutex);
        // check if we can just write to result vector
        if (!result_vector[output_index]) {
          result_vector[output_index].swap(result);
          // we are done here
          AS_LOG_DEBUG << "exiting second critical section and returing"
                       << std::endl;
          return true;
        }
        // get the pointer of the resulut vector
        prior_result.swap(result_vector[output_index]);
        // result_vector[output_index] now holds a nullptr
      }  // end result_mutex
      AS_LOG_DEBUG << "exiting second critical section" << std::endl;
      // perform addition
      result->addInPlace(prior_result);
    }
  };

  // populate the ctxt vector
  run_parallel(conv_func);

  return Ctxt(result_vector, std::shared_ptr<Layout>(layout),
              "conv(" + lhs.getName() + ")");
}

Ctxt BatchLayout::pad(Ctxt& lhs, const xla::PaddingConfig& pad_config,
                      const xla::Shape& new_shape, double pad_value) const {
  AS_LOG_INFO << "BatchLayout pad" << std::endl;
  // compute size of the new shape
  int64_t size = 1;
  for (size_t i = 1; i < new_shape.dimensions().size(); i++) {
    size *= new_shape.dimensions()[i];
  }
  // create return vector and fill it with nullpointers
  AS_LOG_DEBUG << "Creating result vector" << std::endl;
  std::vector<shared_ptr<HECtxt>> res_vec(size, shared_ptr<HECtxt>());

  // copy over the values we need
  std::vector<int64_t> input_index(lhs.shape().size(), 0);
  std::vector<int64_t> target_index(new_shape.rank(), 0);

  auto shark_shape = xla_shape_to_shark_shape(new_shape);

  // create batched shapes
  auto target_shape_batched = shark_shape;
  target_shape_batched[0] = 1;
  auto input_shape_batched = lhs.shape();
  input_shape_batched[0] = 1;
  // Loop through each element of the operand, assign them to the
  // corresponding index of the resulting padded literal.
  auto func = [&](absl::Span<const int64_t> input_index) {
    for (auto i = 0; i < input_index.size(); ++i) {
      // Interior padding occurs logically before edge padding, so in the
      // case of negative edge padding elements are removed from the
      // interior-padded operand.
      target_index[i] =
          pad_config.dimensions(i).edge_padding_low() +
          input_index[i] * (pad_config.dimensions(i).interior_padding() + 1);

      // Account for negative low and high padding: skip assignment if the
      // any target index is out of range.
      if (!(target_index[i] >= 0 &&
            target_index[i] < new_shape.dimensions(i))) {
        return true;
      }
    }
    // flat indices
    size_t target_index_f =
        multi_index_to_flat(target_index, target_shape_batched);
    size_t input_index_f =
        multi_index_to_flat(input_index, input_shape_batched);
    AS_LOG_DEBUG << "Copying value from "
                 << IterablePrintWrapper<absl::Span<const int64_t>>(input_index)
                 << "(" << input_index_f << ") to "
                 << IterablePrintWrapper<absl::Span<const int64_t>>(
                        target_index)
                 << "(" << target_index_f << ")" << std::endl;
    res_vec[target_index_f] = lhs.getValue()[input_index_f]->deepCopy();
    return true;
  };

  std::vector<int64_t> zero_base(lhs.shape().size(), 0);
  std::vector<int64_t> step(lhs.shape().size(), 1);

  // next we cheat a little by creating a shape that has a batch dim of 1
  Shape temp_shape = lhs.shape();
  temp_shape[0] = 1;
  AS_LOG_DEBUG << "Creating dummy shape" << std::endl;
  auto xla_dummy_shape = create_xla_dummy_shape(temp_shape);
  AS_LOG_DEBUG << "Starting iteration:" << std::endl;
  xla::ShapeUtil::ForEachIndex(xla_dummy_shape, zero_base,
                               xla::AsInt64Slice(xla_dummy_shape.dimensions()),
                               step, func);

  // now we need to replace all the null pointers
  AS_LOG_DEBUG << "Creating padding values:" << std::endl;
  std::vector<double> pad_vec{pad_value};
  for (size_t i = 0; i < res_vec.size(); ++i) {
    if (!res_vec[i]) {
      AS_LOG_DEBUG << "Creating padding values at:" << i << "/"
                   << res_vec.size() << std::endl;
      AS_LOG_DEBUG << "context " << (const void*)(lhs.getContext())
                   << std::endl;
      res_vec[i] =
          shared_ptr<HECtxt>(lhs.getContext()->encrypt(pad_vec, "padding"));
    }
  }
  // create and return padded ciphertext
  AS_LOG_DEBUG << "Creating layout" << std::endl;
  auto layout =
      std::shared_ptr<Layout>(createLayout(LAYOUT_TYPE::BATCH, shark_shape));
  return Ctxt(res_vec, layout, "padded ctxt");
}

Ctxt BatchLayout::reshape(Ctxt& lhs, const Shape& shape) const {
  if (lhs.shape()[0] != shape[0]) {
    AS_LOG_ERROR << "can't reshape batch dimension. not implemented yet"
                 << std::endl;
    throw std::runtime_error(
        "can't reshape batch dimension. not implemented yet");
  }
  std::shared_ptr<Layout> layout(createLayout(LAYOUT_TYPE::BATCH, shape));
  // AS_LOG_INFO << "reshaping from " << lhs.layout() << " to "  << *layout
  // << std::endl;
  lhs.updateLayout(layout);
  AS_LOG_INFO << "reshaped to " << lhs.shape() << std::endl;
  return lhs;
};

// template instantiation
template Ctxt BatchLayout::dot_internal<Ctxt, HECtxt>(const Ctxt& one,
                                                      const Ctxt& two) const;
template Ctxt BatchLayout::mat_mult_internal<Ctxt, HECtxt>(
    const Ctxt& one, const Ctxt& two) const;
// template Ctxt BatchLayout::mat_mult_general_internal<Ctxt, HECtxt>(
//     const Ctxt& one, const Ctxt& two) const;

template Ctxt BatchLayout::dot_internal<Ptxt, HEPtxt>(const Ctxt& one,
                                                      const Ptxt& two) const;
template Ctxt BatchLayout::mat_mult_internal<Ptxt, HEPtxt>(
    const Ctxt& one, const Ptxt& two) const;
// template Ctxt BatchLayout::mat_mult_general_internal<Ptxt, HEPtxt>(
//     const Ctxt& one, const Ptxt& two) const;

// register batch layout
static bool batch_init = [] {
  // create factory function
  AS_LOG_DEBUG << "Batch layout registerd" << std::endl;
  auto factory = [](const Shape& shape) { return new BatchLayout(shape); };
  registerLayout(LAYOUT_TYPE::BATCH, factory);
  return true;
}();

}  // namespace aluminum_shark
