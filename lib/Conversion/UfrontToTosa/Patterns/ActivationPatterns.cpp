#include "../Patterns.hpp"
#include "../Util.hpp"

namespace mlir {
namespace ufront {

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

// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
struct GeluHelper {
  static constexpr auto PI = 3.141592653589793;
  static constexpr auto COEFFICIENT = 0.044715;

  // halfPart(x) = 0.5 * x
  static Value halfPart(Value x, OpBuilder& builder) {
    constexpr auto HALF = 0.5;
    constexpr auto SHIFT = 0;

    auto loc = x.getLoc();
    auto type = x.getType();
    auto attr = getDenseFloatAttr(HALF, type, builder);
    auto half = builder.create<tosa::ConstOp>(loc, type, attr);
    return builder.create<tosa::MulOp>(loc, type, x, half, SHIFT);
  }

  // sqrtPart(x) = sqrt(2 / pi)
  static Value sqrtPart(Value x, OpBuilder& builder) {
    constexpr auto HALF = 0.5;

    auto loc = x.getLoc();
    auto type = x.getType();

    auto halfPiAttr = getDenseFloatAttr(PI * HALF, type, builder);
    auto halfPi = builder.create<tosa::ConstOp>(loc, type, halfPiAttr);

    return builder.create<tosa::RsqrtOp>(loc, type, halfPi);
  }

  // powPart(x) = x + 0.044715 * x^3
  static Value powPart(Value x, OpBuilder& builder) {
    constexpr auto CUBE = 3.0;
    constexpr auto SHIFT = 0;

    auto loc = x.getLoc();
    auto type = x.getType();

    auto coAttr = getDenseFloatAttr(COEFFICIENT, type, builder);
    auto cubeAttr = getDenseFloatAttr(CUBE, type, builder);

    auto coefficient = builder.create<tosa::ConstOp>(loc, type, coAttr);
    auto cube = builder.create<tosa::ConstOp>(loc, type, cubeAttr);
    auto pow = builder.create<tosa::PowOp>(loc, type, x, cube);
    auto mul = builder.create<tosa::MulOp>(loc, type, pow, coefficient, SHIFT);

    return builder.create<tosa::AddOp>(loc, type, x, mul);
  }

  static Value gelu(Value x, OpBuilder& builder) {
    constexpr auto ONE = 1.0;
    constexpr auto SHIFT = 0;

    auto loc = x.getLoc();
    auto type = x.getType();

    auto half = halfPart(x, builder);
    auto sqrt = sqrtPart(x, builder);
    auto pow = powPart(x, builder);
    auto mul = builder.create<tosa::MulOp>(loc, type, sqrt, pow, SHIFT);

    auto oneAttr = getDenseFloatAttr(ONE, type, builder);
    auto one = builder.create<tosa::ConstOp>(loc, type, oneAttr);
    auto add = builder.create<tosa::AddOp>(loc, type, one, mul);

    return builder.create<tosa::MulOp>(loc, type, half, add, SHIFT);
  }
};

LogicalResult GeluConverter::matchAndRewrite(GeluOp gelu,
                                             PatternRewriter& rewriter) const {
  rewriter.replaceOp(gelu, GeluHelper::gelu(gelu.getInput(), rewriter));
  return success();
}

LogicalResult SiluConverter::matchAndRewrite(SiluOp silu,
                                             PatternRewriter& rewriter) const {
  auto loc = silu->getLoc();
  auto input = silu.getInput();
  auto type = silu.getType();
  auto shift = rewriter.getI32IntegerAttr(0);

  auto sigmoid = rewriter.create<tosa::SigmoidOp>(loc, type, input);
  rewriter.replaceOpWithNewOp<tosa::MulOp>(silu, type, input, sigmoid, shift);
  return success();
}

LogicalResult SigmoidConverter::matchAndRewrite(
    SigmoidOp sigmoid, PatternRewriter& rewriter) const {
  auto input = sigmoid.getInput();
  auto type = sigmoid.getType();
  rewriter.replaceOpWithNewOp<tosa::SigmoidOp>(sigmoid, type, input);
  return success();
}

Value hardsigmoidPiecewise(Value x, Value geValue, Value leValue,
                           Value elseValue, OpBuilder& builder) {
  constexpr auto UPPER = 3.0;
  constexpr auto LOWER = -3.0;

  auto type = x.getType().cast<ShapedType>();
  auto elemTy = type.getElementType();
  auto loc = x.getLoc();

  auto upper = constantScalar(UPPER, elemTy, builder);
  auto lower = constantScalar(LOWER, elemTy, builder);

  auto condType = RankedTensorType::get(type.getShape(), builder.getI1Type());
  auto ge = builder.create<tosa::GreaterEqualOp>(loc, condType, x, upper);
  auto le = builder.create<tosa::GreaterEqualOp>(loc, condType, lower, x);

  auto selectLe =
      builder.create<tosa::SelectOp>(loc, type, le, leValue, elseValue);
  return builder.create<tosa::SelectOp>(loc, type, ge, geValue, selectLe);
}

LogicalResult HardSigmoidConverter::matchAndRewrite(
    HardSigmoidOp hs, PatternRewriter& rewriter) const {
  auto x = hs.getInput();
  auto type = hs.getType();
  auto elemTy = type.getElementType();
  auto loc = hs->getLoc();

  auto zero = constantScalar(0.0, elemTy, rewriter);
  auto one = constantScalar(1.0, elemTy, rewriter);

  auto oneSixth = constantScalar(1.0 / 6.0, elemTy, rewriter);
  auto oneHalf = constantScalar(0.5, elemTy, rewriter);

  auto shift = rewriter.getI32IntegerAttr(0);
  auto mul = rewriter.create<tosa::MulOp>(loc, type, x, oneSixth, shift);
  auto add = rewriter.create<tosa::AddOp>(loc, type, mul, oneHalf);

  rewriter.replaceOp(hs, hardsigmoidPiecewise(x, one, zero, add, rewriter));
  return success();
}

LogicalResult HardSwishConverter::matchAndRewrite(
    HardSwishOp hs, PatternRewriter& rewriter) const {
  auto x = hs.getInput();
  auto type = hs.getType();
  auto elemTy = type.getElementType();
  auto loc = hs->getLoc();

  auto three = constantScalar(3.0, elemTy, rewriter);

  auto zero = constantScalar(0.0, elemTy, rewriter);
  auto oneSixth = constantScalar(1.0 / 6.0, elemTy, rewriter);

  auto shift = rewriter.getI32IntegerAttr(0);
  auto add = rewriter.create<tosa::AddOp>(loc, type, x, three);
  auto mul = rewriter.create<tosa::MulOp>(loc, type, add, oneSixth, shift);
  auto res = rewriter.create<tosa::MulOp>(loc, type, x, mul, shift);

  rewriter.replaceOp(hs, hardsigmoidPiecewise(x, x, zero, res, rewriter));
  return success();
}

}  // namespace ufront
}  // namespace mlir
