#ifndef ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PTXT_H
#define ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PTXT_H

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/aluminum_shark/base_txt.h"
#include "tensorflow/compiler/plugin/aluminum_shark/he_backend/he_backend.h"

namespace aluminum_shark {

// A class that wraps around a storage object which holds the actual data.
// Needs to be cheap to copy which is why the storage object is held in a shared
// pointer. But this means none of these objects actually exclusively own the
// stored object. this class holds an encoded HEPtxt
//
class Ptxt : public BaseTxt {
 public:
  // Ctor/Dtor
  Ptxt(const xla::Literal& l);
  // Ptxt(const Ptxt& other) = default;
  // Ptxt(Ptxt&& other) = default;
  Ptxt(const xla::Literal& l, std::string name);
  Ptxt(const xla::Literal& l, std::shared_ptr<Layout> layout, std::string name);
  virtual ~Ptxt(){};

  // getters / setters
  const std::vector<std::shared_ptr<HEPtxt>>& getValue() const;
  std::vector<std::shared_ptr<HEPtxt>>& getValue();
  void setValue(std::vector<std::shared_ptr<HEPtxt>>& value_ptr);

  const std::string& getName() const;
  void setName(const std::string& name);

  virtual const Shape& shape() const override;
  const xla::Literal& literal() const { return literal_; };
  // void literal(xla::Literal& l) { literal_ = l; };

  HEContext* context() { return context_; };
  void context(HEContext* c) { context_ = c; };

  // create a deep copy which also creates a copy of the stored object
  Ptxt deepCopy() const;

  bool is_initialized() const;

  // BaseTxt interface
  std::string to_string() const override;

  void updateLayout(std::shared_ptr<Layout> layout) override;
  void updateLayout(std::shared_ptr<Layout> layout, const HEContext* context);
  void updateLayout(LAYOUT_TYPE layout_type, const HEContext* context);

  std::shared_ptr<BaseTxt> operator+(const BaseTxt& other) const override;
  std::shared_ptr<BaseTxt> operator*(const BaseTxt& other) const override;

  BaseTxt& operator+=(const BaseTxt& other) override;
  BaseTxt& operator*=(const BaseTxt& other) override;

  // integer
  std::shared_ptr<BaseTxt> operator+(long other) override;
  std::shared_ptr<BaseTxt> operator*(long other) override;
  Ptxt& operator+=(long other) override;
  Ptxt& operator*=(long other) override;
  // floating point
  std::shared_ptr<BaseTxt> operator+(double other) override;
  std::shared_ptr<BaseTxt> operator*(double other) override;
  Ptxt& operator+=(double other) override;
  Ptxt& operator*=(double other) override;

  // TODO RP: template this
  std::vector<double> decodeDouble() const;
  std::vector<long> decodeLong() const;

 private:
  std::vector<std::shared_ptr<HEPtxt>> value_;
  std::string name_;
  HEContext* context_;
  const xla::Literal& literal_;
  Shape shape_;  // use it when layout is present
};

// convert literal into Ptxt

template <xla::PrimitiveType LiteralT, typename PtxtType>
std::vector<PtxtType> convertLiteralToPtxt(const xla::Literal& literal,
                                           const xla::ShapeIndex& index) {
  auto data = literal.data<
      typename ::xla::primitive_util::PrimitiveTypeToNative<LiteralT>::type>(
      index);
  return std::vector<PtxtType>(data.begin(), data.end());
};

template <class T>
std::vector<T> convertLiteralToPtxt(const xla::Literal& literal) {
  std::vector<T> ptxt;
  xla::ShapeUtil::ForEachSubshape(
      literal.shape(),
      [&](const xla::Shape& subshape, const xla::ShapeIndex& index) -> void {
        if (subshape.IsArray()) {
          AS_LOG_SA << "\tShapeIndex " << index.ToString() << " isArray"
                    << std::endl;
          // convert to plaintext
          switch (literal.shape().element_type()) {
            // int types
            case ::xla::PrimitiveType::PRED: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::PRED, T>(
                  literal, index);
              break;
            }
            case ::xla::PrimitiveType::S8: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::S8, T>(literal,
                                                                       index);
              break;
            }
            case ::xla::PrimitiveType::S16: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::S16, T>(literal,
                                                                        index);
              break;
            }
            case ::xla::PrimitiveType::S32: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::S32, T>(literal,
                                                                        index);
              break;
            }
            case ::xla::PrimitiveType::S64: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::S64, T>(literal,
                                                                        index);
              break;
            }
            case ::xla::PrimitiveType::U8: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::U8, T>(literal,
                                                                       index);
              break;
            }
            case ::xla::PrimitiveType::U16: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::U16, T>(literal,
                                                                        index);
              break;
            }
            case ::xla::PrimitiveType::U32: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::U32, T>(literal,
                                                                        index);
              break;
            }
            case ::xla::PrimitiveType::U64: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::S64, T>(literal,
                                                                        index);
              break;
            }
            // float types
            case ::xla::PrimitiveType::F16: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::F16, T>(literal,
                                                                        index);
              break;
            }
            case ::xla::PrimitiveType::F32: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::F32, T>(literal,
                                                                        index);
              break;
            }
            case ::xla::PrimitiveType::BF16: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::BF16, T>(
                  literal, index);
              break;
            }
            case ::xla::PrimitiveType::F64: {
              ptxt = convertLiteralToPtxt<::xla::PrimitiveType::F64, T>(literal,
                                                                        index);
              break;
            }
            // complex types
            case ::xla::PrimitiveType::C64:
            case ::xla::PrimitiveType::C128:
            default:
              AS_LOG_ERROR << "Error: Unsupported Data Type" << std::endl;
              break;
          }
        } else {
          AS_LOG_INFO << "\tShapeIndex " << index.ToString() << std::endl;
        }
      });
  AS_LOG_INFO << "Created plaintext " << std::endl;
  if (ptxt.size() != 0) {
    return ptxt;
  }
  throw std::runtime_error("invalid data type");
};

}  // namespace aluminum_shark

#endif /* ALUMINUM_SHARK_DEPENDENCIES_TENSORFLOW_TENSORFLOW_COMPILER_PLUGIN_ALUMINUM_SHARK_PTXT_H \
        */
