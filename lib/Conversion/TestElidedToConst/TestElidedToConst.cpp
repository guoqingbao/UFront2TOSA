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

template< class Iter >
void get_uniform_array( Iter start, Iter end, float min, float max, float scale)
{
    static std::random_device rd;    
    static std::mt19937 mte(rd());  
    std::normal_distribution<> dist(min, max);
    std::generate(start, end, [&] () { return dist(mte) * scale; });
}

class ElidedConverter : public OpRewritePattern<ElidedOp> {
  using OpRewritePattern<ElidedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ElidedOp elided,
                                PatternRewriter& rewriter) const override {
    // constexpr auto MAGICNUM = 0.777;
    auto type = elided.getType();
    auto shape = type.getShape();
    auto total = std::accumulate(shape.begin(), shape.end(), 1L,
                                 std::multiplies<int64_t>());

    std::vector<float> values(total);
    get_uniform_array(values.begin(), values.end(), -1.0, 1.0, sqrtf32(2.0/total));

    // for (auto i = 0L; i < total; i++) {
    //   values.emplace_back(APFloat{nc::random::randN<float>()});
    // }

    auto attr = DenseElementsAttr::get(type, llvm::ArrayRef(values));
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(elided, type, attr);
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
