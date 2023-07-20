#include "Conversion/Passes.hpp"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::ufront {

#define GEN_PASS_DEF_FIXTOSAMATMUL
#define GEN_PASS_DECL_FIXTOSAMATMUL
#include "Conversion/Passes.hpp.inc"

class FixTosaMatmulPattern;

class FixTosaMatmul : public impl::FixTosaMatmulBase<FixTosaMatmul> {
  auto runOnOperation() -> void override {
    RewritePatternSet patterns{&getContext()};
    patterns.insert<FixTosaMatmulPattern>(&getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

auto createFixTosaMatmul() -> std::unique_ptr<Pass> {
  return std::make_unique<FixTosaMatmul>();
}

class FixTosaMatmulPattern : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern<tosa::MatMulOp>::OpRewritePattern;

  auto matchAndRewrite(tosa::MatMulOp matmul, PatternRewriter& rewriter) const
      -> LogicalResult override {
    auto lhs = matmul.getA();
    auto rhs = matmul.getB();

    auto lhsTy = cast<TensorType>(lhs.getType());
    auto rhsTy = cast<TensorType>(rhs.getType());

    if (lhsTy.getDimSize(0) == rhsTy.getDimSize(0)) {
      return success();
    }

    // 只考虑了 weight 是 2d tensor 的情况，即 tensor<1xMxNxf32>
    assert(rhsTy.getDimSize(0) == 1);
    auto loc = matmul->getLoc();

    auto squeezedTy = RankedTensorType::get(rhsTy.getShape().take_back(2),
                                            rhsTy.getElementType());
    auto squeezed = rewriter.create<tosa::ReshapeOp>(loc, squeezedTy, rhs,
                                                     squeezedTy.getShape());

    auto empty = rewriter
                     .create<tensor::EmptyOp>(
                         loc,
                         ArrayRef{lhsTy.getDimSize(0), rhsTy.getDimSize(1),
                                  rhsTy.getDimSize(2)},
                         rhsTy.getElementType())
                     .getResult();
    auto broadcasted =
        rewriter
            .create<linalg::BroadcastOp>(loc, squeezed, empty, ArrayRef{0L})
            ->getResult(0);

    rewriter.replaceOpWithNewOp<tosa::MatMulOp>(matmul, matmul.getType(), lhs,
                                                broadcasted);
    return success();
  }
};

}  // namespace mlir::ufront
