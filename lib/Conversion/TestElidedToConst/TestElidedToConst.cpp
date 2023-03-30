#include "Conversion/TestElidedToConst/TestElidedToConst.hpp"

#include "Dialect/Ufront/IR/Ufront.hpp"
// #include "NumCpp/Random/randN.hpp"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>
#include <array>
#include <iterator>
#include <random>
#include <algorithm>
#include <math.h>

namespace mlir {
namespace ufront {

template <class Iter>
void getNormalArray(Iter start, Iter end, float min, float max, float scale) {
  static std::random_device rd;
  static std::mt19937 mte(rd());
  std::normal_distribution<> dist(min, max);
  std::generate(start, end, [&]() { return dist(mte) * scale; });
}

template <typename Iter>
void getUniformArray(Iter start, Iter end, float min, float max, float scale) {
  static std::random_device rd;
  static std::mt19937 mte{rd()};
  std::uniform_real_distribution<> dist{min, max};
  std::generate(start, end, [&]() { return dist(mte) * scale; });
}

Value initWeightForConv2D(ElidedOp elided, OpBuilder& builder) {
  auto type = elided.getType();
  auto inputShapeAttr = elided->getAttrOfType<ArrayAttr>("conv2d_input_shape");

  assert(inputShapeAttr && "requires attribute `conv2d_input_shape`");
  assert(inputShapeAttr.size() == 4 && "`conv2d_input_shape` must be 4D");

  // NCHW
  auto channel = dyn_cast<IntegerAttr>(inputShapeAttr[1]);
  assert(channel && "dims of `conv2d_input_shape` must be integer");
  auto inFeatures = channel.getInt();

  std::vector<float> values(inFeatures);
  getNormalArray(values.begin(), values.end(), -1.0, 1.0,
                 sqrtf32(2.0 / inFeatures));

  auto attr = DenseElementsAttr::get(type, llvm::ArrayRef(values));
  return builder.create<tosa::ConstOp>(elided.getLoc(), type, attr);
}

Value initWeightForLinear(ElidedOp elided, OpBuilder& builder) {
  auto type = elided.getType();
  auto outputShapeAttr =
      elided->getAttrOfType<ArrayAttr>("linear_output_shape");

  assert(outputShapeAttr && "requires attribute `linear_output_shape`");

  // channel first
  auto channel = dyn_cast_or_null<IntegerAttr>(outputShapeAttr[1]);
  assert(channel && "dims of `linear_output_shape` must be integer");
  auto outFeatures = channel.getInt();

  auto range = sqrtf32(1.0 / outFeatures);
  std::vector<float> values(outFeatures);
  getUniformArray(values.begin(), values.end(), -range, range, 1.0);

  auto attr = DenseElementsAttr::get(type, llvm::ArrayRef(values));
  return builder.create<tosa::ConstOp>(elided->getLoc(), type, attr);
}

class ElidedConverter : public OpRewritePattern<ElidedOp> {
  using OpRewritePattern<ElidedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElidedOp elided,
                                PatternRewriter& rewriter) const override {
    // constexpr auto MAGICNUM = 0.777;
    // auto type = elided.getType();
    // auto shape = type.getShape();
    // auto total = std::accumulate(shape.begin(), shape.end(), 1L,
    //                              std::multiplies<int64_t>());

    // std::vector<float> values(total);
    // get_uniform_array(values.begin(), values.end(), -1.0, 1.0,
    // sqrtf32(2.0/total));

    // auto attr = DenseElementsAttr::get(type, llvm::ArrayRef(values));
    // rewriter.replaceOpWithNewOp<tosa::ConstOp>(elided, type, attr);
    auto init = elided->getAttrOfType<StringAttr>("init");
    if (!init) {
      return failure();
    }

    using Fn = function_ref<Value(ElidedOp, PatternRewriter&)>;
    auto fn = StringSwitch<Fn>(init.strref())
                  .Case("conv2d", initWeightForConv2D)
                  .Case("linear", initWeightForLinear)
                  .Default(nullptr);

    if (!fn) {
      return failure();
    }

    rewriter.replaceOp(elided, fn(elided, rewriter));
    return success();
  }
};

class TestConvertElidedToConst
    : public impl::TestConvertElidedToConstBase<TestConvertElidedToConst> {
  void runOnOperation() override {
    auto patterns = RewritePatternSet{&getContext()};
    patterns.add<ElidedConverter>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createTestConvertElidedToConst() {
  return std::make_unique<TestConvertElidedToConst>();
}

}  // namespace ufront
}  // namespace mlir
