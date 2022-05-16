
#include "tensorflow/compiler/plugin/aluminum_shark/base_txt.h"

#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"

namespace aluminum_shark {

const Shape& BaseTxt::shape() { return layout_->shape(); }

const Layout& BaseTxt::layout() { return *layout_; }

void BaseTxt::setLayout(std::shared_ptr<Layout> layout) { layout_ = layout; }

}  // namespace aluminum_shark
