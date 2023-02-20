#include "Patterns.hpp"

#include "Dialect/Ufront/IR/Ufront.hpp"
#include "Util.hpp"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace ufront {

void populateConvertUfrontToTosaPatterns(RewritePatternSet& patterns) {
  // clang-format off
  patterns.add<AddConverter, 
               ReluConverter, 
               FlatConverter, 
               Conv2DConverter,
               BatchnormConverter, 
               LinearConverter, 
               SoftmaxConverter,
               Pool2DConverter>(patterns.getContext());
  // clang-format on
}

SmallVector<int64_t> getIntValueFromArrayAttr(ArrayAttr array) {
  auto valueIt = [](Attribute attr) {
    return attr.cast<IntegerAttr>().getInt();
  };

  auto values = SmallVector<int64_t>{};
  transform(array, std::back_inserter(values), valueIt);
  return values;
}

LogicalResult AddConverter::matchAndRewrite(AddOp add,
                                            PatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<tosa::AddOp>(add, add.getType(), add.getLhs(),
                                           add.getRhs());
  return success();
};

LogicalResult ReluConverter::matchAndRewrite(ReluOp relu,
                                             PatternRewriter& rewriter) const {
  auto maxFp = rewriter.getF32FloatAttr(std::numeric_limits<float>::max());
  auto minFp = rewriter.getF32FloatAttr(0);
  auto maxInt = rewriter.getI64IntegerAttr(std::numeric_limits<int>::max());
  auto minInt = rewriter.getI64IntegerAttr(0);
  rewriter.replaceOpWithNewOp<tosa::ClampOp>(
      relu, relu.getType(), relu.getInput(), minInt, maxInt, minFp, maxFp);
  return success();
}

LogicalResult FlatConverter::matchAndRewrite(FlatOp flat,
                                             PatternRewriter& rewriter) const {
  auto shape = flat.getType().getShape();
  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(flat, flat.getType(),
                                               flat.getInput(), shape);
  return success();
}

LogicalResult Conv2DConverter::matchAndRewrite(
    Conv2DOp conv, PatternRewriter& rewriter) const {
  auto loc = conv->getLoc();
  auto input = conv.getInput();
  auto inTy = input.getType();
  auto output = conv.getOutput();
  auto outTy = output.getType();
  auto elemTy = inTy.getElementType();

  // pad (attribute)
  auto pad = conv.getPadding();
  auto padVals = getIntValueFromArrayAttr(pad);
  auto newPad = rewriter.getDenseI64ArrayAttr(
      {padVals[0], padVals[0], padVals[1], padVals[1]});

  // stride (attribute)
  // auto stride = conv.getStride().cast<DenseI64ArrayAttr>();
  auto stride = conv.getStride();
  auto strideVals = getIntValueFromArrayAttr(stride);
  auto newStride = rewriter.getDenseI64ArrayAttr(strideVals);

  // dilation (attribute)
  auto dilation = rewriter.getDenseI64ArrayAttr({1, 1});

  // input (operand)
  auto newInput = transpose(input, {0, 2, 3, 1}, rewriter);

  // weight (operand)
  auto outShape = outTy.getShape();
  auto kernel = conv.getKernel();
  auto intVal = [](Attribute attr) {
    return attr.cast<IntegerAttr>().getInt();
  };
  auto weightShape = SmallVector<int64_t, 4>{
      outShape[1], intVal(kernel[0]), intVal(kernel[1]), inTy.getDimSize(1)};
  auto weight = rewriter.create<ElidedOp>(loc, weightShape, elemTy);

  // bias (operand)
  auto biasShape = SmallVector<int64_t, 1>{outShape[1]};
  auto biasType = RankedTensorType::get(biasShape, elemTy);
  auto biasAttr = DenseElementsAttr::get(biasType, rewriter.getF32FloatAttr(0));
  auto bias = rewriter.create<tosa::ConstOp>(loc, biasType, biasAttr);

  // result
  auto resShape = SmallVector<int64_t, 4>{outShape[0], outShape[2], outShape[3],
                                          outShape[1]};
  auto resType = RankedTensorType::get(resShape, elemTy);
  auto res = rewriter.create<tosa::Conv2DOp>(loc, resType, newInput, weight,
                                             bias, newPad, newStride, dilation);

  rewriter.replaceOp(conv, transpose(res, {0, 3, 1, 2}, rewriter));

  return success();
}

// y = ((x - E(x)) / (Var(x) + epsilon)) * gamma + beta
// where: E(x) = 0, Var(x) = 1, epsilon = 1e-5, gamma = 1, bata = 0
LogicalResult BatchnormConverter::matchAndRewrite(
    BatchnormOp bn, PatternRewriter& rewriter) const {
  auto loc = bn->getLoc();
  auto epsilonShape = SmallVector<int64_t, 3>{1, 1, 1};
  auto epsilonType = RankedTensorType::get(epsilonShape, rewriter.getF32Type());
  auto epsilonAttr =
      DenseElementsAttr::get(epsilonType, rewriter.getF32FloatAttr(0.00001));
  auto epsilon = rewriter.create<tosa::ConstOp>(loc, epsilonType, epsilonAttr);

  auto x = bn.getInput();
  auto xType = x.getType();
  auto exAttr = DenseElementsAttr::get(xType, rewriter.getF32FloatAttr(0.0));
  auto varxAttr = DenseElementsAttr::get(xType, rewriter.getF32FloatAttr(1.0));
  auto ex = rewriter.create<tosa::ConstOp>(loc, xType, exAttr);
  auto varx = rewriter.create<tosa::ConstOp>(loc, xType, varxAttr);

  auto sub = rewriter.create<tosa::SubOp>(loc, xType, x, ex);
  auto add = rewriter.create<tosa::AddOp>(loc, xType, varx, epsilon);
  auto rsqrt = rewriter.create<tosa::RsqrtOp>(loc, xType, add);

  auto shift = rewriter.getI32IntegerAttr(0);
  rewriter.replaceOpWithNewOp<tosa::MulOp>(bn, xType, sub, rsqrt, shift);

  return success();
}

LogicalResult LinearConverter::matchAndRewrite(
    LinearOp linear, PatternRewriter& rewriter) const {
  auto input = linear.getInput();
  auto inputType = input.getType();
  auto inputShape = inputType.getShape();

  auto reshapeDims = SmallVector<int64_t, 3>{inputShape};
  reshapeDims.insert(reshapeDims.begin(), 1);
  auto reshapeInput = reshape(input, reshapeDims, rewriter);

  auto output = linear.getOutput();
  auto outputShape = output.getType().getShape();

  auto weightShape = SmallVector<int64_t, 3>{1, reshapeDims[2], outputShape[1]};
  auto weight = rewriter.create<ElidedOp>(linear->getLoc(), weightShape,
                                          inputType.getElementType());

  auto mm = matmul(reshapeInput, weight, rewriter);
  rewriter.replaceOp(linear, reshape(mm, outputShape, rewriter));

  return success();
}

LogicalResult SoftmaxConverter::matchAndRewrite(
    SoftmaxOp softmax, PatternRewriter& rewriter) const {
  auto loc = softmax.getLoc();
  auto input = softmax.getInput();

  auto exp = rewriter.create<tosa::ExpOp>(loc, input.getType(), input);
  auto sum = exp.getResult();
  auto sumType = sum.getType();
  auto sumShape = sumType.cast<ShapedType>().getShape();

  for (auto [axis, dim] : enumerate(sumShape)) {
    if (dim == 1) {
      continue;
    }

    sum = reduceSum(sum, axis, rewriter);
  }

  auto rec = rewriter.create<tosa::ReciprocalOp>(loc, sum.getType(), sum);
  auto shift = rewriter.getI32IntegerAttr(0);
  rewriter.replaceOpWithNewOp<tosa::MulOp>(softmax, sumType, exp, rec, shift);

  return success();
}

LogicalResult maxPool2D(Pool2DOp pool, PatternRewriter& rewriter) {
  auto kernel = pool->getAttrOfType<ArrayAttr>("kernel");
  auto stride = pool->getAttrOfType<ArrayAttr>("stride");
  auto padding = pool->getAttrOfType<ArrayAttr>("padding");

  auto kernelVals = getIntValueFromArrayAttr(kernel);
  auto strideVals = getIntValueFromArrayAttr(stride);
  auto paddingVals = getIntValueFromArrayAttr(padding);

  auto kernelAttr = rewriter.getDenseI64ArrayAttr(kernelVals);
  auto strideAttr = rewriter.getDenseI64ArrayAttr(strideVals);
  auto padAttr = rewriter.getDenseI64ArrayAttr(
      {paddingVals[0], paddingVals[0], paddingVals[1], paddingVals[1]});

  rewriter.replaceOpWithNewOp<tosa::MaxPool2dOp>(
      pool, pool.getType(), pool.getInput(), kernelAttr, strideAttr, padAttr);

  return success();
}

LogicalResult adaptivePool2D(Pool2DOp pool, PatternRewriter& rewriter) {
  auto stride = rewriter.getDenseI64ArrayAttr({1, 1});
  auto padding = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});

  auto outSizeAttr = pool->getAttrOfType<ArrayAttr>("output_size");
  auto outSizeVals = SmallVector<int64_t, 2>{};
  transform(outSizeAttr, std::back_inserter(outSizeVals), [](Attribute attr) {
    return attr.dyn_cast<IntegerAttr>().getInt();
  });

  auto transposed = transpose(pool.getInput(), {0, 2, 3, 1}, rewriter);
  auto oldType = transposed.getType().dyn_cast<ShapedType>();
  auto oldShape = oldType.getShape();

  auto kernelVals = SmallVector<int64_t, 2>{};
  kernelVals.emplace_back(oldShape[1] / outSizeVals[0]);
  kernelVals.emplace_back(oldShape[2] / outSizeVals[1]);
  auto kernel = rewriter.getDenseI64ArrayAttr(kernelVals);

  auto newShape = SmallVector<int64_t>{oldShape};
  newShape[1] = outSizeVals[0];
  newShape[2] = outSizeVals[1];
  auto newType = RankedTensorType::get(newShape, oldType.getElementType());

  auto pooled = rewriter.create<tosa::AvgPool2dOp>(
      pool->getLoc(), newType, transposed, kernel, stride, padding);

  rewriter.replaceOp(pool, transpose(pooled, {0, 3, 1, 2}, rewriter));

  return success();
}

struct PoolType {
  constexpr static StringLiteral POOL_MAX = "PoolType.POOL_MAX";
  constexpr static StringLiteral POOL_ADAPTIVE = "PoolType.POOL_ADAPTIVE";
};

LogicalResult Pool2DConverter::matchAndRewrite(
    Pool2DOp pool, PatternRewriter& rewriter) const {
  using Fn = function_ref<LogicalResult(Pool2DOp, PatternRewriter&)>;

  auto poolType = pool.getPoolType();
  auto fn = StringSwitch<Fn>(poolType)
                .Case(PoolType::POOL_MAX, maxPool2D)
                .Case(PoolType::POOL_ADAPTIVE, adaptivePool2D)
                .Default(nullptr);

  if (!fn) {
    return failure();
  }

  return fn(pool, rewriter);
}

}  // namespace ufront
}  // namespace mlir
