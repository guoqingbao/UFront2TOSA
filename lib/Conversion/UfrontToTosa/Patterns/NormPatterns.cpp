#include "../Patterns.hpp"
#include "../Util.hpp"
#include "Conversion/Passes.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace ufront {

Optional<Value> mean(Value tensor, ArrayRef<int64_t> dims, OpBuilder& builder) {
  auto type = tensor.getType().cast<ShapedType>();
  for (auto dim : dims) {
    if (dim < 0 || dim >= type.getRank()) {
      return std::nullopt;
    }
  }

  auto area = 1L;
  auto result = tensor;
  for (auto dim : dims) {
    area *= type.getDimSize(dim);
    result = reduceSum(result, dim, builder);
  }

  auto elemTy = type.getElementType();
  auto cst = constantScalar(1.0 / area, elemTy, builder);

  auto shift = builder.getI32IntegerAttr(0);
  return builder.create<tosa::MulOp>(result.getLoc(), result.getType(), result,
                                     cst, shift);
}

Optional<Value> norm(Value tensor, ArrayRef<int64_t> dims, OpBuilder& builder,
                     double EPS = 0.00001) {
  auto loc = tensor.getLoc();
  auto type = tensor.getType();

  auto shift = builder.getI32IntegerAttr(0);
  auto square = [&](Value x) {
    return builder.create<tosa::MulOp>(loc, type, x, x, shift);
  };

  auto meanX = mean(tensor, dims, builder);
  if (!meanX) {
    return std::nullopt;
  }

  auto meanOfSqrX = mean(square(tensor), dims, builder);
  if (!meanOfSqrX) {
    return std::nullopt;
  }

  auto sqrOfMeanX = square(*meanX);
  auto varX = builder.create<tosa::SubOp>(loc, type, *meanOfSqrX, sqrOfMeanX);

  auto eps = constantScalar(EPS, getElementTypeOrSelf(type), builder);
  auto sub = builder.create<tosa::SubOp>(loc, type, tensor, *meanX);
  auto add = builder.create<tosa::AddOp>(loc, type, varX, eps);
  auto rsqrt = builder.create<tosa::RsqrtOp>(loc, type, add);

  return builder.create<tosa::MulOp>(loc, type, sub, rsqrt, shift);
}

Optional<Value> meanNCHW(Value tensor, OpBuilder& builder) {
  constexpr auto RANK = 4;
  constexpr auto DIMS = std::array<int64_t, 3>{0, 2, 3};

  auto loc = tensor.getLoc();
  auto type = tensor.getType().cast<ShapedType>();
  if (type.getRank() != RANK) {
    tensor.getDefiningOp()->emitError() << "Rank of tensor must be 4\n";
    return std::nullopt;
  }

  auto total = 1UL;
  auto reduced = tensor;
  for (auto dim : DIMS) {
    total *= dim;
    reduced = reduceSum(reduced, dim, builder);
  }

  auto attr = getDenseFloatAttr(1.0 / total, type, builder);
  auto reciprocal = builder.create<tosa::ConstOp>(loc, type, attr);

  auto shift = builder.getI32IntegerAttr(0);
  return builder.create<tosa::MulOp>(loc, type, reduced, reciprocal, shift);
}

Optional<Value> normNCHW(Value tensor, OpBuilder& builder,
                         double EPS = 0.00001) {
  constexpr auto RANK = 4;

  auto type = tensor.getType().cast<ShapedType>();
  if (type.getRank() != RANK) {
    tensor.getDefiningOp()->emitError() << "Rank of tensor must be 4\n";
    return std::nullopt;
  }
  auto loc = tensor.getLoc();

  auto mean = *meanNCHW(tensor, builder);
  auto shift = builder.getI32IntegerAttr(0);
  auto sqr = builder.create<tosa::MulOp>(loc, type, tensor, tensor, shift);
  // E[x^2]
  auto meanOfSqr = *meanNCHW(sqr, builder);
  // E[x]^2
  auto sqrOfMean = builder.create<tosa::MulOp>(loc, type, mean, mean, shift);
  auto var = builder.create<tosa::SubOp>(loc, type, meanOfSqr, sqrOfMean);

  auto eps = constantScalar(EPS, type.getElementType(), builder);
  auto add = builder.create<tosa::AddOp>(loc, type, var, eps);
  auto rsqrt = builder.create<tosa::RsqrtOp>(loc, type, add);
  auto sub = builder.create<tosa::SubOp>(loc, type, tensor, mean);
  return builder.create<tosa::MulOp>(loc, type, sub, rsqrt, shift);
}

// y = ((x - E(x)) / (Var(x) + epsilon)) * gamma + beta
// where: E(x) = 0, Var(x) = 1, epsilon = 1e-5, gamma = 1, beta = 0
LogicalResult BatchNormConverter::matchAndRewrite(
    BatchNormOp bn, PatternRewriter& rewriter) const {
  auto eps = bn.getEps().convertToDouble();

  auto normRes = norm(bn.getInput(), {0, 2, 3}, rewriter, eps);
  if (!normRes) {
    return failure();
  }

  rewriter.replaceOp(bn, *normRes);
  return success();
}

LogicalResult LayerNormConverter::matchAndRewrite(
    LayerNormOp ln, PatternRewriter& rewriter) const {
  auto eps = ln.getEps().convertToDouble();

  auto rank = ln.getInput().getType().getRank();
  auto normShape = ln.getNormalizedShape();
  auto dims = SmallVector<int64_t>{};

  auto normRes = norm(ln.getInput(), dims, rewriter, eps);
  if (!normRes) {
    return failure();
  }

  for (auto i = 0U; i < normShape.size(); i++) {
    dims.emplace_back(rank - 1 - i);
  }

  rewriter.replaceOp(ln, *normRes);
  return success();
}

}  // namespace ufront
}  // namespace mlir
