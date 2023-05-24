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


// %output = tf.FusedBatchNorm(%x, %scale, %offset, %mean, %variance) {epsilon, data_format, is_training}


// assert(data_format == 'NHWC')
// assert(is_training == false)

// %epsilon_const = tosa.CONST() {value={epsilon}}

// %op1 = tosa.SUB(%x, %bmean)
// %op2 = tosa.ADD(%variance, %epsilon_const)
// %op3 = tosa.RSQRT(%op2)
// %op4 = tosa.MUL(%op1, %op3)
// %op5 = tosa.MUL(%op4, %scale)
// %output = tosa.ADD(%op5, %offset)
Optional<Value> batchnorm(OpBuilder& builder, Value tensor, ArrayRef<int64_t> dims, Value weight, Value bias, 
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
  auto add = builder.create<tosa::AddOp>(loc, varX.getType(), varX, eps);
  auto rsqrt = builder.create<tosa::RsqrtOp>(loc, add.getType(), add);

  auto mul = builder.create<tosa::MulOp>(loc, type, sub, rsqrt, shift);
  auto weightProd = builder.create<tosa::MulOp>(loc, type, mul, weight, shift);
  return builder.create<tosa::AddOp>(loc, type, weightProd, bias);
}

Optional<Value> batchnorm_mean_var(OpBuilder& builder, Value tensor, ArrayRef<int64_t> dims, 
                                  Value weight, Value bias, Value mean, Value variance, double EPS = 0.00001) {
  auto loc = tensor.getLoc();
  auto type = tensor.getType();

  auto shift = builder.getI32IntegerAttr(0);

  auto eps = constantScalar(EPS, getElementTypeOrSelf(type), builder);
  auto sub = builder.create<tosa::SubOp>(loc, type, tensor, mean);
  auto add = builder.create<tosa::AddOp>(loc, variance.getType(), variance, eps);
  auto rsqrt = builder.create<tosa::RsqrtOp>(loc, add.getType(), add);

  auto mul = builder.create<tosa::MulOp>(loc, type, sub, rsqrt, shift);
  auto weightProd = builder.create<tosa::MulOp>(loc, type, mul, weight, shift);
  return builder.create<tosa::AddOp>(loc, type, weightProd, bias);
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

LogicalResult batchnorm_option(PatternRewriter& rewriter, BatchNormOp& bn, Value weight, Value bias, Value mean, Value variance, double eps) {
  if (mean && variance) {
    auto normRes = batchnorm_mean_var(rewriter, bn.getInput(), {0, 2, 3}, weight, bias, mean, variance, eps);
    if (normRes) {
      rewriter.replaceOp(bn, *normRes);
      return success();
    }
  } else {
    auto normRes = batchnorm(rewriter, bn.getInput(), {0, 2, 3}, weight, bias, eps);
    if (normRes) {
      rewriter.replaceOp(bn, *normRes);
      return success();
    }
  }
  return failure();
}

// y = ((x - E(x)) / (Var(x) + epsilon)) * gamma + beta
// where: E(x) = 0, Var(x) = 1, epsilon = 1e-5, gamma = 1, beta = 0
LogicalResult BatchNormConverter::matchAndRewrite(
    BatchNormOp bn, PatternRewriter& rewriter) const {
  auto eps = bn.getEps().convertToDouble();
  auto weight = bn.getWeight();
  auto bias = bn.getBias();
  auto mean = bn.getMean();
  auto variance = bn.getVariance();
  if (weight && bias) {
    return batchnorm_option(rewriter, bn, weight, bias, mean, variance, eps);
  } else {
    auto outTy = bn.getType();
    // auto elemTy = outTy.getElementType();
    auto shape = SmallVector<int64_t, 4>{1, outTy.getDimSize(1), 1, 1};
    auto newTy = outTy.clone(shape);

    std::vector<float> values(outTy.getDimSize(1), 1.0);
    auto attr = DenseElementsAttr::get(newTy, llvm::ArrayRef(values));
    auto weight1 = rewriter.create<tosa::ConstOp>(bn.getLoc(), newTy, attr);

    std::vector<float> values1(outTy.getDimSize(1), 0.0);
    auto attr1 = DenseElementsAttr::get(newTy, llvm::ArrayRef(values1));
    auto bias1 = rewriter.create<tosa::ConstOp>(bn.getLoc(), newTy, attr1);
    return batchnorm_option(rewriter, bn, weight1, bias1, mean, variance, eps);
  }
}

LogicalResult LayerNormConverter::matchAndRewrite(
    LayerNormOp ln, PatternRewriter& rewriter) const {
  auto eps = ln.getEps().convertToDouble();

  auto rank = ln.getInput().getType().getRank();
  auto normShape = ln.getNormalizedShape();
  auto dims = SmallVector<int64_t>{};
  for (auto i = 0U; i < normShape.size(); i++) {
    dims.emplace_back(rank - 1 - i);
  }

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

LogicalResult MeanConverter::matchAndRewrite(MeanOp op,
                                             PatternRewriter& rewriter) const {
  auto input = op.getInput();
  auto dims = getIntValueFromArrayAttr(op.getDims());
  auto keepdims = op.getKeepdims();

  auto result = mean(input, dims, rewriter);
  if (!result) {
    return failure();
  }

  if (!keepdims) {
    auto type = input.getType();
    auto newShape = SmallVector<int64_t>{};

    for (auto i : llvm::seq(0L, type.getRank())) {
      if (llvm::find(dims, i) != dims.end()) {
        continue;
      }
      newShape.emplace_back(type.getDimSize(i));
    }

    result = reshape(*result, newShape, rewriter);
  }

  rewriter.replaceOp(op, *result);
  return success();
}

}  // namespace ufront
}  // namespace mlir
