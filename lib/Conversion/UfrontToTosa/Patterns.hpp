#include "Conversion/UfrontToTosa/UfrontToTosa.hpp"
#include "Dialect/Ufront/IR/Ufront.hpp"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace ufront {

void populateConvertUfrontToTosaPatterns(RewritePatternSet& patterns);

class AddConverter : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AddOp add,
                                PatternRewriter& rewriter) const override;
};

class ReluConverter : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReluOp relu,
                                PatternRewriter& rewriter) const override;
};

class FlatConverter : public OpRewritePattern<FlatOp> {
  using OpRewritePattern<FlatOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(FlatOp flat,
                                PatternRewriter& rewriter) const override;
};

class Conv2DConverter : public OpRewritePattern<Conv2DOp> {
  using OpRewritePattern<Conv2DOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(Conv2DOp conv,
                                PatternRewriter& rewriter) const override;
};

class BatchnormConverter : public OpRewritePattern<BatchnormOp> {
  using OpRewritePattern<BatchnormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BatchnormOp bn,
                                PatternRewriter& rewriter) const override;
};

class LinearConverter : public OpRewritePattern<LinearOp> {
  using OpRewritePattern<LinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LinearOp linear,
                                PatternRewriter& rewriter) const override;
};

class SoftmaxConverter : public OpRewritePattern<SoftmaxOp> {
  using OpRewritePattern<SoftmaxOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SoftmaxOp softmax,
                                PatternRewriter& rewriter) const override;
};

class Pool2DConverter : public OpRewritePattern<Pool2DOp> {
  using OpRewritePattern<Pool2DOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(Pool2DOp pool,
                                PatternRewriter& rewriter) const override;
};

}  // namespace ufront
}  // namespace mlir
