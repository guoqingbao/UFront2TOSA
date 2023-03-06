#include "Patterns.hpp"

#include <functional>
#include <numeric>

#include "Dialect/Ufront/IR/Ufront.hpp"
#include "Util.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace ufront {

void populateConvertUfrontToTosaPatterns(RewritePatternSet& patterns) {
  // clang-format off
  patterns.add<AddConverter, 
               ReluConverter, 
               FlatConverter, 
               Conv2DConverter,
               BatchNormConverter, 
               LinearConverter, 
               SoftmaxConverter,
               Pool2DConverter,
               ReshapeConverter,
               ConcatConverter,
               DropoutConverter,
               TransposeConverter,
               ExpandConverter,
               GeluConverter,
               SliceConverter,
               LayerNormConverter,
               MultiplyConverter,
               SigmoidConverter,
               SiluConverter,
               HardSigmoidConverter,
               HardSwishConverter,
               BatchMatmulConverter,
               MaskedFillConverter,
               MultiheadAttentionConverter>(patterns.getContext());
  // clang-format on
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
  auto pad = conv.getPad();
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
  auto weight = rewriter.create<ParameterOp>(loc, weightShape, elemTy);

  // bias (operand)
  auto biasShape = SmallVector<int64_t, 1>{outShape[1]};
  auto biasType = RankedTensorType::get(biasShape, elemTy);
  auto biasAttr = DenseElementsAttr::get(biasType, rewriter.getF32FloatAttr(0));
  auto bias = rewriter.create<tosa::ConstOp>(loc, biasType, biasAttr);

  // result
  auto resShape = SmallVector<int64_t, 4>{outShape[0], outShape[2], outShape[3],
                                          outShape[1]};
  auto resType = RankedTensorType::get(resShape, elemTy);

  Value res;

  if (auto group = conv.getGroups(); group == 1) {
    res = rewriter.create<tosa::Conv2DOp>(loc, resType, newInput, weight, bias,
                                          newPad, newStride, dilation);
  } else if (group == static_cast<uint64_t>(inTy.getDimSize(1))) {
    res = rewriter.create<tosa::DepthwiseConv2DOp>(
        loc, resType, newInput, weight, bias, newPad, newStride, dilation);
  } else {
    return failure();
  }

  rewriter.replaceOp(conv, transpose(res, {0, 3, 1, 2}, rewriter));

  return success();
}

Value norm(Value x, Value mean, Value var, Value eps, Value weight, Value bias,
           OpBuilder& builder) {
  auto loc = x.getLoc();
  auto type = x.getType();

  auto sub = builder.create<tosa::SubOp>(loc, type, x, mean);
  auto add = builder.create<tosa::AddOp>(loc, type, var, eps);
  auto rsqrt = builder.create<tosa::RsqrtOp>(loc, type, add);
  auto shift = builder.getI32IntegerAttr(0);
  auto mul = builder.create<tosa::MulOp>(loc, type, sub, rsqrt, shift);
  auto weightProd = builder.create<tosa::MulOp>(loc, type, mul, weight, shift);
  return builder.create<tosa::AddOp>(loc, type, weightProd, bias);
}

Optional<Value> meanNCHW(Value tensor, OpBuilder& builder) {
  constexpr auto RANK = 4;
  constexpr auto DIMS = std::array<uint64_t, 3>{0, 2, 3};

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

Optional<Value> normNCHW(Value tensor, OpBuilder& builder) {
  constexpr auto RANK = 4;
  constexpr auto EPS = 0.00001;
  // constexpr auto WEIGHT = 1.0;
  // constexpr auto BIAS = 0.0;

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
  auto norm = normNCHW(bn.getInput(), rewriter);
  if (!norm) {
    return failure();
  }
  rewriter.replaceOp(bn, *norm);

  return success();
}

LogicalResult LinearConverter::matchAndRewrite(
    LinearOp linear, PatternRewriter& rewriter) const {
  auto input = linear.getInput();
  auto inTy = input.getType();
  auto outTy = linear.getType();
  auto rank = inTy.getRank();
  auto elemTy = inTy.getElementType();

  auto shape = SmallVector<int64_t, 3>{inTy.getDimSize(rank - 1),
                                       outTy.getDimSize(rank - 1)};
  if (2 == rank) {
    shape.insert(shape.begin(), 1);
    auto inShape = SmallVector<int64_t>{inTy.getShape()};
    inShape.insert(inShape.begin(), 1);
    input = reshape(input, inShape, rewriter);
  } else if (3 == rank) {
    shape.insert(shape.begin(), inTy.getDimSize(0));
  }

  auto weight = rewriter.create<ParameterOp>(linear->getLoc(), shape, elemTy);
  auto result = matmul(input, weight, rewriter);
  rewriter.replaceOp(linear, reshape(result, outTy.getShape(), rewriter));
  return success();
}

LogicalResult maxPool2D(Pool2DOp pool, PatternRewriter& rewriter) {
  auto kernel = pool->getAttrOfType<ArrayAttr>("kernel");
  auto stride = pool->getAttrOfType<ArrayAttr>("stride");
  auto padding = pool->getAttrOfType<ArrayAttr>("pad");

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
  constexpr static StringLiteral POOL_MAX = "POOL_MAX";
  constexpr static StringLiteral POOL_ADAPTIVE = "POOL_ADAPTIVE";
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

LogicalResult ReshapeConverter::matchAndRewrite(
    ReshapeOp reshape, PatternRewriter& rewriter) const {
  auto shapeAttr = reshape.getShape();
  auto shapeVals = getIntValueFromArrayAttr(shapeAttr);

  auto type = reshape.getType();
  auto input = reshape.getInput();
  auto reshapeAttr = rewriter.getDenseI64ArrayAttr(shapeVals);

  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(reshape, type, input,
                                               reshapeAttr);
  return success();
}

LogicalResult ConcatConverter::matchAndRewrite(
    ConcatOp concat, PatternRewriter& rewriter) const {
  auto axis = concat.getAxis();
  auto type = concat.getType();
  rewriter.replaceOpWithNewOp<tosa::ConcatOp>(concat, type,
                                              concat->getOperands(), axis);
  return success();
}

LogicalResult DropoutConverter::matchAndRewrite(
    DropoutOp dropout, PatternRewriter& rewriter) const {
  auto rate = dropout.getRate();
  srand(dropout.getSeed());

  if (rate.isZero()) {
    rewriter.replaceOp(dropout, dropout.getInput());
  } else if (rate.isExactlyValue(1.0)) {
    auto type = dropout.getType();
    auto attr = getDenseFloatAttr(1.0, type, rewriter);
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(dropout, type, attr);
  } else {
    auto type = dropout.getType();
    auto shape = type.getShape();
    auto length =
        reduce(shape.begin(), shape.end(), 1L, std::multiplies<int64_t>());

    auto flags = SmallVector<bool>{};
    flags.reserve(length);
    auto range = static_cast<int>(1.0 / rate.convertToDouble());

    for (auto i = 0L; i < length; i++) {
      flags.emplace_back((rand() % range) != 0);
    }

    auto loc = dropout->getLoc();
    auto flagType = RankedTensorType::get(shape, rewriter.getI1Type());
    auto flagAttr = DenseElementsAttr::get(flagType, flags);
    auto flagCst = rewriter.create<tosa::ConstOp>(loc, flagType, flagAttr);
    auto zero = constantScalar(0.0, type.getElementType(), rewriter);

    rewriter.replaceOpWithNewOp<tosa::SelectOp>(dropout, type, flagCst,
                                                dropout.getInput(), zero);
    return success();
  }

  return success();
}

LogicalResult TransposeConverter::matchAndRewrite(
    TransposeOp op, PatternRewriter& rewriter) const {
  auto permsAttr = op.getPerms();
  auto permsVals = getIntValueFromArrayAttr(permsAttr);
  rewriter.replaceOp(op, transpose(op.getInput(), permsVals, rewriter));
  return success();
}

LogicalResult ExpandConverter::matchAndRewrite(
    ExpandOp expand, PatternRewriter& rewriter) const {
  auto sizesAttr = expand.getSizes();
  auto sizesVals = getIntValueFromArrayAttr(sizesAttr);

  auto input = expand.getInput();
  auto inTy = input.getType();
  auto inShape = inTy.getShape();
  auto outTy = expand.getType();

  auto cstSize = SmallVector<int64_t>{inShape};
  for (auto [i, dim] : enumerate(inShape)) {
    if (dim == -1) {
      continue;
    }

    cstSize[i] = dim;
  }

  auto cstType = RankedTensorType::get(cstSize, inTy.getElementType());
  auto cstAttr = getDenseFloatAttr(1.0, cstType, rewriter);
  auto cst = rewriter.create<tosa::ConstOp>(expand->getLoc(), cstType, cstAttr);

  rewriter.replaceOpWithNewOp<tosa::MulOp>(expand, outTy, input, cst, 0);
  return success();
}

LogicalResult SliceConverter::matchAndRewrite(SliceOp op,
                                              PatternRewriter& rewriter) const {
  auto offsets = SmallVector<int64_t>{};
  auto sizes = SmallVector<int64_t>{};
  auto strides = SmallVector<int64_t>{};

  auto input = op.getInput();
  auto inTy = input.getType();
  auto slices = op->getAttr("slices").dyn_cast_or_null<ArrayAttr>();
  if (!slices) {
    return failure();
  }

  for (auto [i, slice] : llvm::enumerate(slices)) {
    if (auto intAttr = slice.dyn_cast<IntegerAttr>(); intAttr) {
      offsets.emplace_back(intAttr.getInt());
      sizes.emplace_back(1);
      strides.emplace_back(1);
      continue;
    }

    auto array = slice.cast<ArrayAttr>();
    auto valueFn = [array](size_t index, int64_t defaultValue) {
      if (array.size() <= index || array[index].isa<StringAttr>()) {
        return defaultValue;
      }
      return array[index].cast<IntegerAttr>().getInt();
    };
    offsets.emplace_back(valueFn(0, 0));
    sizes.emplace_back(valueFn(1, inTy.getDimSize(sizes.size())));
    strides.emplace_back(valueFn(2, 1));
  }

  while (offsets.size() != static_cast<size_t>(inTy.getRank())) {
    offsets.emplace_back(0);
    sizes.emplace_back(inTy.getDimSize(sizes.size()));
    strides.emplace_back(1);
  }

  auto outTy = op.getType();
  rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
      op, outTy, input, ValueRange{}, ValueRange{}, ValueRange{},
      rewriter.getDenseI64ArrayAttr(offsets),
      rewriter.getDenseI64ArrayAttr(sizes),
      rewriter.getDenseI64ArrayAttr(strides));

  return failure();
}

LogicalResult LayerNormConverter::matchAndRewrite(
    LayerNormOp ln, PatternRewriter& rewriter) const {
  auto x = ln.getInput();
  auto type = x.getType();

  auto shape = SmallVector<int64_t, 4>{type.getShape()};
  auto rank = shape.size();
  while (shape.size() != 4) {
    shape.insert(shape.begin(), 1);
  }

  auto nchw = reshape(x, shape, rewriter);
  auto cnhw = transpose(nchw, {1, 0, 2, 3}, rewriter);
  auto norm = normNCHW(cnhw, rewriter);

  if (!norm) {
    return failure();
  }
  auto transposed = transpose(*norm, {1, 0, 2, 3}, rewriter);
  auto transposedShape = transposed.getType().cast<ShapedType>().getShape();
  rewriter.replaceOp(
      ln, reshape(transposed, transposedShape.take_back(rank), rewriter));
  return success();
}

LogicalResult MaskedFillConverter::matchAndRewrite(
    MaskedFillOp mf, PatternRewriter& rewriter) const {
  constexpr auto ZERO = 0.0;

  auto loc = mf->getLoc();
  auto mask = mf.getMask();
  auto type = mf.getType();
  auto elemTy = type.getElementType();
  auto shape = type.getShape();
  auto condType = RankedTensorType::get(shape, rewriter.getI1Type());

  auto zero = constantScalar(ZERO, elemTy, rewriter);
  auto cond = rewriter.create<tosa::EqualOp>(loc, condType, mask, zero);

  auto value = mf.getValue().convertToDouble();
  auto valueTensor = constantScalar(value, elemTy, rewriter);

  rewriter.replaceOpWithNewOp<tosa::SelectOp>(mf, type, cond, valueTensor,
                                              mf.getInput());
  return success();
}

}  // namespace ufront
}  // namespace mlir
