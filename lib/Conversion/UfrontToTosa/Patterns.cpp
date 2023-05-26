#include "Patterns.hpp"

#include <functional>
#include <numeric>
#include <typeindex>

#include "Dialect/Ufront/IR/Ufront.hpp"
#include "Util.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
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
               MultiheadAttentionConverter,
               ChunkConverter,
               MeanConverter,
               ParameterConverter,
               SaddConverter,
               SmultiplyConverter,
               SplitConverter,
               SubtractConverter,
               MatmulConverter,
               StrueDivConverter>(patterns.getContext());
  // clang-format on
}

LogicalResult FlatConverter::matchAndRewrite(FlatOp flat,
                                             PatternRewriter& rewriter) const {
  auto shape = flat.getType().getShape();
  rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(flat, flat.getType(),
                                               flat.getInput(), shape);
  return success();
}

Value lowerToConv2D(Conv2DOp conv, OpBuilder& builder) {
  auto loc = conv->getLoc();
  auto input = conv.getInput();
  auto inTy = input.getType();
  auto output = conv.getOutput();
  auto outTy = output.getType();
  auto elemTy = inTy.getElementType();

  // pad (attribute)
  auto pad = conv.getPad();
  auto padVals = getIntValueFromArrayAttr(pad);
  auto newPad = builder.getDenseI64ArrayAttr(
      {padVals[0], padVals[0], padVals[1], padVals[1]});

  // stride (attribute)
  // auto stride = conv.getStride().cast<DenseI64ArrayAttr>();
  auto stride = conv.getStride();
  auto strideVals = getIntValueFromArrayAttr(stride);
  auto newStride = builder.getDenseI64ArrayAttr(strideVals);

  // dilation (attribute)
  auto dilation = builder.getDenseI64ArrayAttr({1, 1});

  // input (operand)
  auto newInput = transpose(input, {0, 2, 3, 1}, builder);

  auto outShape = outTy.getShape();

  // bias (operand)
  auto biasShape = SmallVector<int64_t, 1>{outShape[1]};
  auto biasType = RankedTensorType::get(biasShape, elemTy);
  auto biasAttr = DenseElementsAttr::get(biasType, builder.getF32FloatAttr(0));
  auto bias = builder.create<tosa::ConstOp>(loc, biasType, biasAttr);

  // result
  auto resShape = SmallVector<int64_t, 4>{outShape[0], outShape[2], outShape[3],
                                          outShape[1]};
  auto resType = RankedTensorType::get(resShape, elemTy);

  auto weight = conv.getWeight();
  // weight (operand)
  if (!weight) {
    auto kernel = conv.getKernel();
    auto intVal = [](Attribute attr) {
      return attr.cast<IntegerAttr>().getInt();
    };
    auto weightShape = SmallVector<int64_t, 4>{
        outShape[1], intVal(kernel[0]), intVal(kernel[1]), inTy.getDimSize(1)};

    auto weight = builder.create<ElidedOp>(loc, weightShape, elemTy);
    weight->setAttr("init", builder.getStringAttr("conv2d"));
    weight->setAttr("output_shape", builder.getI64ArrayAttr(outTy.getShape()));
    auto res = builder.create<tosa::Conv2DOp>(loc, resType, newInput, weight,
                                            bias, newPad, newStride, dilation);
    return transpose(res, {0, 3, 1, 2}, builder);
  }
  else {
    auto weight1 = transpose(weight, {0, 2, 3, 1}, builder);
    Value bias1 = conv.getBias();
    auto res = builder.create<tosa::Conv2DOp>(loc, resType, newInput, weight1,
                                            bias1?bias1:bias, newPad, newStride, dilation);
    return transpose(res, {0, 3, 1, 2}, builder);
  }
}

// TODO: refactor
Value lowerToDepthwiseConv2D(Conv2DOp conv, OpBuilder& builder) {
  auto loc = conv->getLoc();
  auto input = conv.getInput();
  auto inTy = input.getType();
  auto output = conv.getOutput();
  auto outTy = output.getType();
  auto elemTy = inTy.getElementType();

  // pad (attribute)
  auto pad = conv.getPad();
  auto padVals = getIntValueFromArrayAttr(pad);
  auto newPad = builder.getDenseI64ArrayAttr(
      {padVals[0], padVals[0], padVals[1], padVals[1]});

  // stride (attribute)
  // auto stride = conv.getStride().cast<DenseI64ArrayAttr>();
  auto stride = conv.getStride();
  auto strideVals = getIntValueFromArrayAttr(stride);
  auto newStride = builder.getDenseI64ArrayAttr(strideVals);

  // dilation (attribute)
  auto dilation = builder.getDenseI64ArrayAttr({1, 1});

  // input (operand)
  auto newInput = transpose(input, {0, 2, 3, 1}, builder);

  auto outShape = outTy.getShape();

  // bias (operand)
  auto biasShape = SmallVector<int64_t, 1>{outShape[1]};
  auto biasType = RankedTensorType::get(biasShape, elemTy);
  auto biasAttr = DenseElementsAttr::get(biasType, builder.getF32FloatAttr(0));
  auto bias = builder.create<tosa::ConstOp>(loc, biasType, biasAttr);

  // result
  auto resShape = SmallVector<int64_t, 4>{outShape[0], outShape[2], outShape[3],
                                          outShape[1]};
  auto resType = RankedTensorType::get(resShape, elemTy);

  // weight (operand)
  auto weight = conv.getWeight();
  if (!weight) {
    auto kernel = conv.getKernel();
    auto intVal = [](Attribute attr) {
      return attr.cast<IntegerAttr>().getInt();
    };
    auto weightShape = SmallVector<int64_t, 4>{
        intVal(kernel[0]), intVal(kernel[1]), inTy.getDimSize(1), 1};
    auto weight = builder.create<ElidedOp>(loc, weightShape, elemTy);
    weight->setAttr("init", builder.getStringAttr("conv2d"));
    weight->setAttr("output_shape", builder.getI64ArrayAttr(outTy.getShape()));
    auto res = builder.create<tosa::DepthwiseConv2DOp>(
        loc, resType, newInput, weight, bias, newPad, newStride, dilation);
    return transpose(res, {0, 3, 1, 2}, builder);
  } else {
      auto weight1 = transpose(weight, {2, 3, 0, 1}, builder);
      auto res = builder.create<tosa::DepthwiseConv2DOp>(
        loc, resType, newInput, weight1, bias, newPad, newStride, dilation);
      return transpose(res, {0, 3, 1, 2}, builder);
  }
}

LogicalResult Conv2DConverter::matchAndRewrite(
    Conv2DOp conv, PatternRewriter& rewriter) const {
  auto group = conv.getGroups();
  auto inTy = conv.getInput().getType();

  if (group == static_cast<uint64_t>(inTy.getDimSize(1))) {
    rewriter.replaceOp(conv, lowerToDepthwiseConv2D(conv, rewriter));
  } else {
    rewriter.replaceOp(conv, lowerToConv2D(conv, rewriter));
  }

  return success();
}

// Value norm(Value x, Value mean, Value var, Value eps, Value weight, Value bias,
//            OpBuilder& builder) {
//   auto loc = x.getLoc();
//   auto type = x.getType();

//   auto sub = builder.create<tosa::SubOp>(loc, type, x, mean);
//   auto add = builder.create<tosa::AddOp>(loc, type, var, eps);
//   auto rsqrt = builder.create<tosa::RsqrtOp>(loc, type, add);
//   auto shift = builder.getI32IntegerAttr(0);
//   auto mul = builder.create<tosa::MulOp>(loc, type, sub, rsqrt, shift);
//   auto weightProd = builder.create<tosa::MulOp>(loc, type, mul, weight, shift);
//   return builder.create<tosa::AddOp>(loc, type, weightProd, bias);
// }

LogicalResult LinearConverter::matchAndRewrite(
    LinearOp linear, PatternRewriter& rewriter) const {
  auto input = linear.getInput();
  auto inTy = input.getType();
  auto outTy = linear.getType();
  auto rank = inTy.getRank();
  auto elemTy = inTy.getElementType();

  auto shape = SmallVector<int64_t, 3>{inTy.getDimSize(rank - 1),
                                       outTy.getDimSize(rank - 1)};

  if (rank < 3) {
    shape.insert(shape.begin(), 1);
    auto inShape = SmallVector<int64_t>{inTy.getShape()};
    while (inShape.size() != 3) {
      inShape.insert(inShape.begin(), 1);
    }
    input = reshape(input, inShape, rewriter);
  } else {
    shape.insert(shape.begin(), inTy.getShape()[rank - 3]);
    input = reshape(input, inTy.getShape().take_back(3), rewriter);
  }

  auto weight = linear.getWeight();
  if (weight) {
    weight = transpose(weight, {1, 0}, rewriter);
    weight = reshape(weight, shape, rewriter);
    auto result = matmul(input, weight, rewriter);
    auto bias = linear.getBias();
    if (bias) {
      auto biased = rewriter.create<tosa::AddOp>(result.getLoc(), result.getType(), result, bias);
      rewriter.replaceOp(linear, reshape(biased, outTy.getShape(), rewriter));
    } else {
      rewriter.replaceOp(linear, reshape(result, outTy.getShape(), rewriter));
    }
  } else {
    auto weight = rewriter.create<ElidedOp>(linear->getLoc(), shape, elemTy);
    weight->setAttr("init", rewriter.getStringAttr("linear"));
    weight->setAttr("output_shape", rewriter.getI64ArrayAttr(outTy.getShape()));
    auto result = matmul(input, weight, rewriter);
    rewriter.replaceOp(linear, reshape(result, outTy.getShape(), rewriter));
  }
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

  auto input = pool.getInput();
  auto transposed = transpose(input, {0, 2, 3, 1}, rewriter);
  auto oldResType = pool.getType();
  auto oldResShape = oldResType.getShape();

  SmallVector<int64_t> newResShape = {oldResShape[0], oldResShape[2],
                                      oldResShape[3], oldResShape[1]};
  auto newResType =
      RankedTensorType::get(newResShape, oldResType.getElementType());

  auto newRes = rewriter.create<tosa::MaxPool2dOp>(
      pool->getLoc(), newResType, transposed, kernelAttr, strideAttr, padAttr);
  rewriter.replaceOp(pool, transpose(newRes, {0, 3, 1, 2}, rewriter));

  return success();
}

// TODO: refactor
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

LogicalResult avgPool2D(Pool2DOp pool, PatternRewriter& rewriter) {
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

  auto input = pool.getInput();
  auto transposed = transpose(input, {0, 2, 3, 1}, rewriter);
  auto oldResType = pool.getType();
  auto oldResShape = oldResType.getShape();

  SmallVector<int64_t> newResShape = {oldResShape[0], oldResShape[2],
                                      oldResShape[3], oldResShape[1]};
  auto newResType =
      RankedTensorType::get(newResShape, oldResType.getElementType());

  auto newRes = rewriter.create<tosa::AvgPool2dOp>(
      pool->getLoc(), newResType, transposed, kernelAttr, strideAttr, padAttr);
  rewriter.replaceOp(pool, transpose(newRes, {0, 3, 1, 2}, rewriter));

  return success();
}

struct PoolType {
  constexpr static StringLiteral POOL_MAX = "POOL_MAX";
  constexpr static StringLiteral POOL_ADAPTIVE = "POOL_ADAPTIVE";
  constexpr static StringLiteral POOL_AVG = "POOL_AVG";
};

LogicalResult Pool2DConverter::matchAndRewrite(
    Pool2DOp pool, PatternRewriter& rewriter) const {
  using Fn = function_ref<LogicalResult(Pool2DOp, PatternRewriter&)>;

  auto poolType = pool.getPoolType();
  auto fn = StringSwitch<Fn>(poolType)
                .Case(PoolType::POOL_MAX, maxPool2D)
                .Case(PoolType::POOL_ADAPTIVE, adaptivePool2D)
                .Case(PoolType::POOL_AVG, avgPool2D)
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

tensor::ExtractSliceOp processSlicesAttr(SliceOp op,
                                         PatternRewriter& rewriter) {
  auto offsets = SmallVector<int64_t>{};
  auto sizes = SmallVector<int64_t>{};
  auto strides = SmallVector<int64_t>{};

  auto input = op.getInput();
  auto inTy = input.getType();
  auto slices = op->getAttr("slices").dyn_cast_or_null<ArrayAttr>();

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

  auto outTy = RankedTensorType::get(sizes, inTy.getElementType());
  auto extract = rewriter.create<tensor::ExtractSliceOp>(
      op->getLoc(), outTy, input, ValueRange{}, ValueRange{}, ValueRange{},
      rewriter.getDenseI64ArrayAttr(offsets),
      rewriter.getDenseI64ArrayAttr(sizes),
      rewriter.getDenseI64ArrayAttr(strides));
  return extract;
}

tensor::ExtractSliceOp processNoSlicesAttr(SliceOp op,
                                           PatternRewriter& rewriter) {
  auto axis = op->getAttrOfType<ArrayAttr>("axis");
  auto end = op->getAttrOfType<ArrayAttr>("end");
  auto start = op->getAttrOfType<ArrayAttr>("start");

  assert(axis && "requires attribute `axis`");
  assert(end && "requires attribute `end`");
  assert(start && "requries attribute `start`");
  assert(axis.size() == start.size() && axis.size() == end.size() &&
         "size mismatched");

  using Tri = std::tuple<int64_t, int64_t, int64_t>;
  DenseMap<int64_t, Tri> map;
  for (auto [a, e, s] : llvm::zip(axis, end, start)) {
    auto axisVal = a.dyn_cast<IntegerAttr>().getInt();
    auto endVal = e.dyn_cast<IntegerAttr>().getInt();
    auto startVal = s.dyn_cast<IntegerAttr>().getInt();
    map[axisVal] = std::make_tuple(axisVal, startVal, endVal);
  }

  auto input = op.getInput();
  auto type = input.getType();
  auto rank = type.getRank();

  SmallVector<int64_t> offsets, sizes, strides;
  for (auto i = 0L; i < rank; ++i) {
    if (map.find(i) == map.end()) {
      offsets.emplace_back(0);
      sizes.emplace_back(type.getDimSize(i));
      strides.emplace_back(1);
    } else {
      auto [_, start, end] = map[i];
      offsets.emplace_back(start);
      sizes.emplace_back(end - start);
      strides.emplace_back(1);
    }
  }

  auto outTy = RankedTensorType::get(sizes, type.getElementType());
  auto extract = rewriter.create<tensor::ExtractSliceOp>(
      op->getLoc(), outTy, input, ValueRange{}, ValueRange{}, ValueRange{},
      rewriter.getDenseI64ArrayAttr(offsets),
      rewriter.getDenseI64ArrayAttr(sizes),
      rewriter.getDenseI64ArrayAttr(strides));
  return extract;
}

LogicalResult SliceConverter::matchAndRewrite(SliceOp op,
                                              PatternRewriter& rewriter) const {
  auto slicesAttr = op->getAttr("slices").dyn_cast_or_null<ArrayAttr>();
  tensor::ExtractSliceOp slice = slicesAttr ? processSlicesAttr(op, rewriter)
                                            : processNoSlicesAttr(op, rewriter);
  rewriter.replaceOp(op, reshape(slice, op.getType().getShape(), rewriter));
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

void split(Value input, uint64_t axis, TypeRange types, OpBuilder& builder,
           SmallVector<Value>& results) {
  results.reserve(types.size());

  auto inTy = cast<TensorType>(input.getType());
  auto offsets = SmallVector<int64_t>{};
  auto sizes = SmallVector<int64_t>{};
  auto strides = SmallVector<int64_t>{};

  for (auto i : llvm::seq(0L, inTy.getRank())) {
    offsets.emplace_back(0);
    sizes.emplace_back(inTy.getDimSize(i));
    strides.emplace_back(1);
  }

  for (auto type : types) {
    sizes[axis] = type.cast<ShapedType>().getDimSize(axis);
    auto slice = builder.create<tensor::ExtractSliceOp>(
        builder.getUnknownLoc(), type, input, ValueRange{}, ValueRange{},
        ValueRange{}, builder.getDenseI64ArrayAttr(offsets),
        builder.getDenseI64ArrayAttr(sizes),
        builder.getDenseI64ArrayAttr(strides));
    results.emplace_back(slice);
    offsets[axis] += sizes[axis];
  }
}

LogicalResult ChunkConverter::matchAndRewrite(ChunkOp op,
                                              PatternRewriter& rewriter) const {
  auto axis = op.getAxis();
  auto input = op.getInput();
  SmallVector<Value> results;
  split(input, axis, op.getResultTypes(), rewriter, results);
  rewriter.replaceOp(op, results);
  return success();
}

LogicalResult SplitConverter::matchAndRewrite(SplitOp op,
                                              PatternRewriter& rewriter) const {
  auto axis = op.getAxis();
  auto input = op.getInput();
  SmallVector<Value> results;
  split(input, axis, op.getResultTypes(), rewriter, results);
  rewriter.replaceOp(op, results);
  return success();
}

}  // namespace ufront
}  // namespace mlir
