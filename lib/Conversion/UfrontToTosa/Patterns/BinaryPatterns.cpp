#include "../Patterns.hpp"

namespace mlir {
namespace ufront {

LogicalResult AddConverter::matchAndRewrite(AddOp add,
                                            PatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<tosa::AddOp>(add, add.getType(), add.getLhs(),
                                           add.getRhs());
  return success();
};

LogicalResult MultiplyConverter::matchAndRewrite(
    MultiplyOp multiply, PatternRewriter& rewriter) const {
  auto lhs = multiply.getLhs();
  auto rhs = multiply.getRhs();
  auto type = multiply.getType();
  auto shift = rewriter.getI32IntegerAttr(0);
  rewriter.replaceOpWithNewOp<tosa::MulOp>(multiply, type, lhs, rhs, shift);
  return success();
}

LogicalResult BatchMatmulConverter::matchAndRewrite(
    BatchMatmulOp bmm, PatternRewriter& rewriter) const {
  auto type = bmm.getType();
  auto lhs = bmm.getLhs();
  auto rhs = bmm.getRhs();
  rewriter.replaceOpWithNewOp<tosa::MatMulOp>(bmm, type, lhs, rhs);
  return success();
}

}  // namespace ufront
}  // namespace mlir
