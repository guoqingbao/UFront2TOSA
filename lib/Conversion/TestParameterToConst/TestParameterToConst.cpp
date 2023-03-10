#include "Conversion/TestParameterToConst/TestParameterToConst.hpp"

#include "Dialect/Ufront/IR/Ufront.hpp"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ufront {

class ParameterConverter : public OpRewritePattern<ParameterOp> {
  using OpRewritePattern<ParameterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParameterOp parameter,
                                PatternRewriter& rewriter) const override {
    constexpr auto MAGICNUM = 0.777;
    auto type = parameter.getType();
    auto attr =
        DenseElementsAttr::get(type, rewriter.getF32FloatAttr(MAGICNUM));
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(parameter, type, attr);
    return success();
  }
};

class TestConvertParameterToConst
    : public impl::TestConvertParameterToConstBase<
          TestConvertParameterToConst> {
  void runOnOperation() override {
    auto patterns = RewritePatternSet{&getContext()};
    patterns.add<ParameterConverter>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

std::unique_ptr<Pass> createTestConvertParameterToConst() {
  return std::make_unique<TestConvertParameterToConst>();
}

}  // namespace ufront
}  // namespace mlir
