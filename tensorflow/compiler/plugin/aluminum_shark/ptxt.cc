#include "tensorflow/compiler/plugin/aluminum_shark/ptxt.h"

#include "tensorflow/compiler/plugin/aluminum_shark/ctxt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

namespace aluminum_shark {

// constructors

Ptxt::Ptxt(const xla::Literal& l)
    : literal_(l), shape_{xla_shape_to_shark_shape(literal_.shape())} {}

Ptxt::Ptxt(const xla::Literal& l, std::string name)
    : literal_(l),
      name_(name),
      shape_{xla_shape_to_shark_shape(literal_.shape())} {}

Ptxt::Ptxt(const xla::Literal& l, std::shared_ptr<Layout> layout,
           std::string name)
    : BaseTxt(layout),
      literal_(l),
      name_(name),
      shape_{xla_shape_to_shark_shape(literal_.shape())} {}

// create a deep copy which also creates a copy of stored object
Ptxt Ptxt::deepCopy() const {
  AS_LOG("creating deep copy of: " + name_);
  Ptxt copy = *this;
  // create a copy of the stored object
  std::vector<std::shared_ptr<HEPtxt>> heptxt_copy;
  for (auto heptxt : value_) {
    heptxt_copy.push_back(std::shared_ptr<HEPtxt>(heptxt->deepCopy()));
  }
  copy.setValue(heptxt_copy);
  // copy layout
  copy.setLayout(std::shared_ptr<Layout>(copy.layout().deepCopy()));
  return copy;
}

bool Ptxt::is_initialized() const { return value_.size() != 0; }

// getters and setters

const std::vector<std::shared_ptr<HEPtxt>>& Ptxt::getValue() const {
  return value_;
}

std::vector<std::shared_ptr<HEPtxt>>& Ptxt::getValue() { return value_; }

const std::string& Ptxt::getName() const { return name_; }
void Ptxt::setName(const std::string& name) { name_ = name; }

void Ptxt::setValue(std::vector<std::shared_ptr<HEPtxt>>& value_ptrs) {
  AS_LOG_S << "setting HEPtxts for " << name_ << std::endl;
  value_ = value_ptrs;
}

// BaseTxt interface

std::string Ptxt::to_string() const {
  // if (is_initialized()) {
  //   return "Plaintext is not initialized.";
  // }
  // TODO: get more info
  return name_;
}

void Ptxt::updateLayout(LAYOUT_TYPE layout_type, const HEContext* context) {
  std::shared_ptr<Layout> layout(
      createLayout(layout_type, xla_shape_to_shark_shape(literal_.shape())));
  updateLayout(layout, context);
}

void Ptxt::updateLayout(std::shared_ptr<Layout> layout,
                        const HEContext* context) {
  updateLayout(layout);
  AS_LOG_INFO << "Clearing ptxts " << std::endl;
  value_.clear();
  AS_LOG_INFO << "Checking scheme type " << std::endl;
  AS_LOG_INFO << "scheme type is " << context->scheme() << std::endl;
  if (context->scheme() == HE_SCHEME::CKKS) {
    AS_LOG_INFO << "CKKS layout " << std::endl;
    std::vector<double> vec = convertLiteralToPtxt<double>(literal_);
    AS_LOG_INFO << "Converted literal to Ptxt " << vec.size() << " items"
                << std::endl;
    if (log_large_vectors()) {
      AS_LOG_DEBUG << vec << std::endl;
    }
    auto vec_with_layout(layout->layout_vector(vec));
    AS_LOG_INFO << "layed out vector" << std::endl;
    for (const auto& v : vec_with_layout) {
      if (log_large_vectors()) {
        AS_LOG_DEBUG << v << std::endl;
      }
      // TODO RP: maybe move here
      value_.push_back(std::shared_ptr<HEPtxt>(context->createPtxt(v)));
    }
  } else if (context->scheme() == HE_SCHEME::BFV) {
    std::vector<long> vec = convertLiteralToPtxt<long>(literal_);
    layout->layout_vector(vec);
    auto vec_with_layout(layout->layout_vector(vec));
    for (const auto& v : vec_with_layout) {
      // TODO RP: maybe move here
      value_.push_back(std::shared_ptr<HEPtxt>(context->createPtxt(v)));
    }
  } else {
    AS_LOG_CRITICAL << "unexpected scheme type" << std::endl;
    throw std::runtime_error("unexpected scheme type");
  }
  AS_LOG_DEBUG << "Plaintext layedout" << std::endl;
}

void Ptxt::updateLayout(std::shared_ptr<Layout> layout) {
  if (layout_) {
    AS_LOG_INFO << "updating layout for " << name_ << " from " << *layout_
                << " to " << *layout << std::endl;
  } else {
    AS_LOG_INFO << "updating layout for " << name_ << " to " << *layout
                << std::endl;
  }
  layout_ = layout;
  AS_LOG_INFO << "layout updated" << std::endl;
  // AS_LOG_S << "number of values in the plaintext: " << value_.size()
  //          << std::endl;
  // const HEContext* context = value_[0]->getContext();
  // AS_LOG_S << "got context: " << context->to_string() << " @"
  //          << reinterpret_cast<const void*>(context) << std::endl;
  // if (context->scheme() == HE_SCHEME::CKKS) {
  //   AS_LOG_S << "CKKS layout update " << std::endl;
  //   std::vector<std::vector<double>> ptxt_with_layout =
  //       layout->layout_vector(decodeDouble());
  //   value_.clear();
  //   for (const auto& ptxt : ptxt_with_layout) {
  //     std::shared_ptr<HEPtxt> ptxt_ptr(
  //         std::shared_ptr<HEPtxt>(context->encode(ptxt)));
  //     value_.push_back(ptxt_ptr);
  //   }
  // } else if (context->scheme() == HE_SCHEME::BFV) {
  //   std::vector<std::vector<long>> ptxt_with_layout =
  //       layout->layout_vector(decodeLong());
  //   value_.clear();
  //   for (const auto& ptxt : ptxt_with_layout) {
  //     std::shared_ptr<HEPtxt> ptxt_ptr(
  //         std::shared_ptr<HEPtxt>(context->encode(ptxt)));
  //     value_.push_back(ptxt_ptr);
  //   }
  // } else {
  //   AS_LOG_S << "unsopported scheme encountered in updating plaintext layout:
  //   "
  //            << context->scheme() << std::endl;
  //   throw std::runtime_error(
  //       "unsopported scheme encountered in updating plaintext layout");
  // }
  // layout_ = layout;
}

const Shape& Ptxt::shape() const {
  if (layout_) {
    return BaseTxt::shape();
  }
  return shape_;
}

std::shared_ptr<BaseTxt> Ptxt::operator+(const BaseTxt& other) const {
  // need to handle the ctxt seperatly. we can't use inplace operations here
  // since the result will be a ctxt
  try {
    const Ctxt& ctxt = dynamic_cast<const Ctxt&>(other);
    return ctxt + *this;
  } catch (const std::bad_cast& e) {
    /* this is ok. ignore */
  }
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr += other;
  return ptr;
}

std::shared_ptr<BaseTxt> Ptxt::operator*(const BaseTxt& other) const {
  // need to handle the ctxt seperatly. we can't use inplace operations here
  // since the result will be a ctxt
  try {
    const Ctxt& ctxt = dynamic_cast<const Ctxt&>(other);
    return ctxt * *this;
  } catch (const std::bad_cast& e) {
    /* this is ok. ignore */
  }
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr *= other;
  return ptr;
}

BaseTxt& Ptxt::operator+=(const BaseTxt& other) {
  AS_LOG_CRITICAL << "there should not be any operations on plaintext"
                  << std::endl;
  throw std::runtime_error("there should not be any operations on plaintext");
}

BaseTxt& Ptxt::operator*=(const BaseTxt& other) {
  AS_LOG_CRITICAL << "there should not be any operations on plaintext"
                  << std::endl;
  throw std::runtime_error("there should not be any operations on plaintext");
}

// integer
std::shared_ptr<BaseTxt> Ptxt::operator+(long other) {
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr += other;
  return ptr;
}

std::shared_ptr<BaseTxt> Ptxt::operator*(long other) {
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr += other;
  return ptr;
}

Ptxt& Ptxt::operator+=(long other) {
  AS_LOG_CRITICAL << "there should not be any operations on plaintext"
                  << std::endl;
  throw std::runtime_error("there should not be any operations on plaintext");
}

Ptxt& Ptxt::operator*=(long other) {
  AS_LOG_CRITICAL << "there should not be any operations on plaintext"
                  << std::endl;
  throw std::runtime_error("there should not be any operations on plaintext");
}

// floating point
std::shared_ptr<BaseTxt> Ptxt::operator+(double other) {
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr += other;
  return ptr;
}

std::shared_ptr<BaseTxt> Ptxt::operator*(double other) {
  // create a copy
  auto ptr = std::make_shared<Ptxt>(deepCopy());
  // call in place operator
  *ptr *= other;
  return ptr;
}

Ptxt& Ptxt::operator+=(double other) {
  AS_LOG_CRITICAL << "there should not be any operations on plaintext"
                  << std::endl;
  throw std::runtime_error("there should not be any operations on plaintext");
}

Ptxt& Ptxt::operator*=(double other) {
  AS_LOG_CRITICAL << "there should not be any operations on plaintext"
                  << std::endl;
  throw std::runtime_error("there should not be any operations on plaintext");
}

// TODO RP: template this
std::vector<double> Ptxt::decodeDouble() const {
  AS_LOG_S << "Decoding double " << std::endl;
  std::vector<std::vector<double>> decodings;
  for (const auto& heptxt : value_) {
    decodings.push_back(heptxt->getContext()->decodeDouble(heptxt.get()));
  }
  if (layout_) {
    AS_LOG_DEBUG << "layout is " << layout() << std::endl;
  } else {
    AS_LOG_DEBUG << "no layout " << std::endl;
  }
  std::vector<double> vec = layout().reverse_layout_vector(decodings);
  if (log(AS_DEBUG)) {
    AS_LOG_DEBUG << "Decoded long. Values:  " << vec << std::endl;
  }
  AS_LOG_SA << "number of values: " << vec.size() << std::endl;
  return vec;
}

// TODO RP: template this
std::vector<long> Ptxt::decodeLong() const {
  std::vector<std::vector<long>> decodings;
  for (const auto& heptxt : value_) {
    decodings.push_back(heptxt->getContext()->decodeLong(heptxt.get()));
  }
  std::vector<long> vec = layout().reverse_layout_vector(decodings);
  AS_LOG_S << "Decoded long. Values: [ ";
  if (log()) {
    aluminum_shark::stream_vector(vec);
  }
  AS_LOG_SA << "number of values: " << vec.size() << std::endl;
  return vec;
}

}  // namespace aluminum_shark
