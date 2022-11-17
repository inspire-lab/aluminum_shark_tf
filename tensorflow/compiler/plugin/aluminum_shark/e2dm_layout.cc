
#include "tensorflow/compiler/plugin/aluminum_shark/e2dm_layout.h"

#include <algorithm>

#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/ptxt.h"

namespace aluminum_shark {

void E2DMLayout::init() {
  AS_LOG_INFO << "Initialzing E2DM layout. shape: ";
  if (log()) {
    stream_vector(shape_);
  }

  // all the data is packed into one ciphertext
  axis_0_ = 1;
  axis_1_ = size_;
}

std::pair<size_t, size_t> E2DMLayout::get_layout_index(size_t i) const {
  return std::pair<size_t, size_t>(0, i);
}

LAYOUT_TYPE E2DMLayout::type() const { return LAYOUT_TYPE::E2DM; }
Layout* E2DMLayout::deepCopy() const { return new E2DMLayout(*this); };

// matrix and vector operations

// dot is the general entry point
Ctxt E2DMLayout::dot(const Ctxt& one, const Ptxt& two) const {
  AS_LOG_INFO << "Perfroming E2DM dot " << std::endl;
  // matrix multiplication
  auto& lhs_v = one.getValue();
  AS_LOG_DEBUG << "number of ctxts on the left hand side " << lhs_v.size()
               << std::endl;
  HECtxt* lhs_hectxt = lhs_v[0].get();

  AS_LOG_INFO << "Dot shapes " << one.shape() << std::endl;
  Shape two_shape = two.shape();
  AS_LOG_INFO << " x " << std::endl;
  AS_LOG_INFO << two_shape << std::endl;
  // get plaintext values
  // todo, need to support longs
  std::vector<double> rhs_v = convertLiteralToPtxt<double>(two.literal());
  Layout* rhs_layout = createLayout(LAYOUT_TYPE::E2DM, two.shape());
  rhs_layout->init();
  rhs_v = rhs_layout->layout_vector(rhs_v)[0];
  delete rhs_layout;

  auto context = one.getContext();
  // holds the result of the first linear transform and at the end it will hold
  // the computation result
  HECtxt* c_0 = nullptr;

  // square case
  if (one.shape().size() == 2 && one.shape()[0] == one.shape()[1]) {
    AS_LOG_DEBUG << "e2dm square matrix multiplication" << std::endl;

    // compute linear transformation sigma from the paper
    size_t d = one.shape()[0];  // dimension of the matrix
    size_t n = d * d;

    for (size_t i = 0; i < 2 * d; ++i) {
      int k = i - d;  // -d < k < k;
      // create diagonal vector
      std::vector<double> u(n, 0);
      if (k >= 0) {
        for (size_t l = 0; l < n; ++l) {
          int temp = l - d * k;
          if (temp >= 0 && temp < d - k) {
            u[l] = 1;
          }
        }
      } else {
        for (size_t l = 0; l < n; ++l) {
          int temp = l - (d + k) * d;
          if (temp >= -k && temp < d) {
            u[l] = 1;
          }
        }
      }
      HEPtxt* ptxt_u = context->createPtxt(u);

      // compute the sum
      if (c_0 == nullptr) {
        // first iteration
        AS_LOG_DEBUG << "rotating k=" << k << " " << lhs_hectxt->to_string()
                     << std::endl;
        c_0 = lhs_hectxt->rotate(k);
        AS_LOG_DEBUG << "rotated " << c_0->to_string() << std::endl;
        c_0->multInPlace(ptxt_u);
        AS_LOG_DEBUG << "masked " << c_0->to_string() << std::endl;
      } else {
        AS_LOG_DEBUG << "rotating k=" << k << " " << lhs_hectxt->to_string()
                     << std::endl;
        HECtxt* temp_ctxt = lhs_hectxt->rotate(k);
        AS_LOG_DEBUG << "rotated " << temp_ctxt->to_string() << std::endl;
        temp_ctxt->multInPlace(ptxt_u);
        AS_LOG_DEBUG << "masked " << temp_ctxt->to_string() << std::endl;
        c_0->addInPlace(temp_ctxt);
        AS_LOG_DEBUG << "added " << c_0->to_string() << std::endl;
        // we are done with temp_ctxt
        delete temp_ctxt;
      }
    }

    AS_LOG_DEBUG << "column shifiting the lhs ctxt" << std::endl;
    // column shift the c_0 d-1 times
    std::vector<HECtxt*> ctxts(d - 1,
                               nullptr);  // holds the column shifted ctxts
    // compute the row shifting diagonal vectors
    for (size_t k = 1; k < d; ++k) {
      // create 2 diagonal vectors
      std::vector<double> v_0(n, 0);
      std::vector<double> v_1(n, 0);
      for (size_t l = 0; l < n; ++l) {
        int lmd = l % d;
        if (lmd >= 0 && lmd < d - k) {
          v_0[l] = 1;
        }
        if (lmd >= d - k && lmd < d) {
          v_1[l] = 1;
        }
      }

      // encode diagonla into plaintext, rotate c0 and mult by plaintext
      HEPtxt* v_0p = context->createPtxt(v_0);
      AS_LOG_DEBUG << "rotating k=" << k << " " << c_0->to_string()
                   << std::endl;
      HECtxt* first = c_0->rotate(k);
      AS_LOG_DEBUG << "rotated " << c_0->to_string() << std::endl;
      first->multInPlace(v_0p);
      AS_LOG_DEBUG << "masked " << first->to_string() << std::endl;
      delete v_0p;

      HEPtxt* v_1p = context->createPtxt(v_1);
      AS_LOG_DEBUG << "rotating k=" << static_cast<signed>(k - d) << " "
                   << c_0->to_string() << std::endl;
      HECtxt* second = c_0->rotate(static_cast<signed>(k - d));
      AS_LOG_DEBUG << "rotated " << c_0->to_string() << std::endl;
      second->multInPlace(v_1p);
      AS_LOG_DEBUG << "masked " << second->to_string() << std::endl;
      delete v_1p;

      // add up the ciphertexts and save result
      first->addInPlace(second);
      AS_LOG_DEBUG << "added " << first->to_string() << std::endl;
      ctxts[k - 1] = first;

      // clean up
      delete second;
    }

    AS_LOG_DEBUG << "column shifting complete " << std::endl;

    AS_LOG_DEBUG << "transforming the rhs " << std::endl;
    // linear transformation on the rhs
    // the permutation on the right is e(A)_i,j = A_i+j,j (all indecies are mod
    // d)
    std::vector<double> p_0(n, 0);
    for (size_t i = 0; i < d; ++i) {
      for (size_t j = 0; j < d; ++j) {
        // index to write to
        size_t w_idx = i * d + j;
        // index to read from
        size_t r_idx = ((i + j) % d) * d + j;
        p_0[w_idx] = rhs_v[r_idx];
      }
    }

    // row shift p_0 d-1 times
    // ψ(A)i, j = Ai+1, j
    std::vector<std::vector<double>> ptxts(d - 1, std::vector<double>(n, 0));
    for (size_t k = 1; k < d; ++k) {
      for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
          // index to write to
          size_t w_idx = i * d + j;
          // index to read from
          size_t r_idx = ((i + j + k) % d) * d + j;
          ptxts[k - 1][w_idx] = rhs_v[r_idx];
        }
      }
    }

    AS_LOG_DEBUG << "rhs transform complete " << std::endl;

    // accumulate the result in c_0
    std::cout
        << "###########################################\n######################"
           "##"
           "###################\n###########################################\n"
        << std::endl;
    AS_LOG_DEBUG << "accumulating results " << std::endl;
    HEPtxt* p_0_enc = context->createPtxt(p_0);
    c_0->multInPlace(p_0_enc);
    AS_LOG_DEBUG << "multiplied " << c_0->to_string() << std::endl;
    delete p_0_enc;

    AS_LOG_DEBUG << "no ctxts: " << ctxts.size() << std::endl;
    for (size_t i = 0; i < ctxts.size(); ++i) {
      AS_LOG_DEBUG << "creating ptxt " << std::endl;
      HEPtxt* p_i = context->createPtxt(ptxts[i]);
      AS_LOG_DEBUG << "getting ctxt " << std::endl;
      HECtxt* c_i = ctxts[i];
      AS_LOG_DEBUG << "got ctxt " << (void*)c_i << " " << c_i->to_string()
                   << std::endl;
      c_i->multInPlace(p_i);
      AS_LOG_DEBUG << "multiplied " << c_i->to_string() << std::endl;
      c_0->addInPlace(c_i);
      AS_LOG_DEBUG << "added " << c_0->to_string() << std::endl;
      // cleanup
      delete p_i;
      delete c_i;
      ctxts[i] = nullptr;
    }
    AS_LOG_DEBUG << "accumulation complete " << std::endl;
    // cleanup
    for (auto ptr : ctxts) {
      if (ptr) {
        delete ptr;
      }
    }
    ctxts.clear();
  } else {
    AS_LOG_DEBUG << "e2dm rectuganlar matrix multiplication" << std::endl;
  }

  // create result objects
  // compute the result shape
  Shape result_shape{one.shape()[0], two.shape()[1]};
  AS_LOG_INFO << "result shape: " << result_shape << std::endl;
  // create result layout
  std::shared_ptr<Layout> result_layout(
      createLayout(LAYOUT_TYPE::E2DM, result_shape));

  std::stringstream result_name;
  result_name << one.getName() << " X " << two.getName();
  std::shared_ptr<HECtxt> temp(c_0);

  return Ctxt(std::vector<std::shared_ptr<HECtxt>>{temp}, result_layout,
              result_name.str());
}

Ctxt E2DMLayout::dot(const Ctxt& one, const Ctxt& two) const {
  AS_LOG_CRITICAL << "E2DMLayout::dot(const Ctxt& one, const Ctxt& two) const "
                     "not implemented yet"
                  << std::endl;
  throw std::runtime_error("not implemented");
};
// Matrix multplication
Ctxt E2DMLayout::mat_mult(const Ctxt& one, const Ptxt& two) const {
  throw std::runtime_error("not implemented");
}
Ctxt E2DMLayout::mat_mult(const Ctxt& one, const Ctxt& two) const {
  AS_LOG_CRITICAL
      << "E2DMLayout::mat_mult(const Ctxt& one, const Ctxt& two) const "
         "not implemented yet"
      << std::endl;
  throw std::runtime_error("not implemented");
};
// More general matrix multplication for hihger dimensional matrices
// see: https://www.tensorflow.org/xla/operation_semantics#dotgeneral, and
// https://en.wikipedia.org/wiki/Tensor_contraction
Ctxt E2DMLayout::mat_mult_general(const Ctxt& one, const Ptxt& two) const {
  AS_LOG_CRITICAL
      << "E2DMLayout::mat_mult_general(const Ctxt& one, const Ptxt& two) const "
         "not implemented yet"
      << std::endl;
  throw std::runtime_error("not implemented");
}
Ctxt E2DMLayout::mat_mult_general(const Ctxt& one, const Ctxt& two) const {
  AS_LOG_CRITICAL << "broadcast needs to be done on plaintext" << std::endl;
  throw std::runtime_error("not implemented");
};

Ctxt E2DMLayout::convolution(const Ctxt& lhs, const Ptxt& rhs,
                             xla::HloInstruction* hlo) const {
  AS_LOG_CRITICAL << "E2DMLayout::convolution(const Ctxt& lhs, const Ptxt& "
                     "rhs, xla::HloInstruction* hlo) const "
                     "not implemented yet"
                  << std::endl;
  throw std::runtime_error("not implemented");
}

Ctxt E2DMLayout::reshape(Ctxt& lhs, const Shape& shape) const { return lhs; };
// private:
// template <class T, class U>
// Ctxt dot_internal(const Ctxt& one, const T& two) const;

// template <class T, class U>
// Ctxt mat_mult_internal(const Ctxt& one, const T& two) const;

// template <class T, class U>
// Ctxt mat_mult_general_internal(const Ctxt& one, const T& two) const;
// }

// register  layout
static bool init = [] {
  AS_LOG_DEBUG << "E2DM layout registerd" << std::endl;
  std::cout << "E2DM layout registerd" << std::endl;
  // create factory function
  auto factory = [](const Shape& shape) { return new E2DMLayout(shape); };
  registerLayout(LAYOUT_TYPE::E2DM, factory);
  return true;
}();

}  // namespace aluminum_shark
