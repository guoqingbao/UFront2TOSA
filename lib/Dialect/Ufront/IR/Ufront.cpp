#include "Dialect/Ufront/IR/Ufront.hpp"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#define GET_OP_CLASSES
#include "Dialect/Ufront/IR/Ufront.cpp.inc"
#include "Dialect/Ufront/IR/UfrontDialect.cpp.inc"

namespace mlir {
namespace ufront {

void UfrontDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Ufront/IR/Ufront.cpp.inc"
      >();
}

void ElidedOp::build(OpBuilder& builder, OperationState& state,
                     ArrayRef<int64_t> shape, Type elementType) {
  auto type = RankedTensorType::get(shape, elementType);
  return build(builder, state, type);
}

LogicalResult LinearOp::verify() {
  auto inTy = getInput().getType();

  if (inTy.getRank() != 2 && inTy.getRank() != 3) {
    emitError() << "Inputs of `ufront.linear` must be rank 2 or rank 3\n";
    return failure();
  }

  return success();
}

LogicalResult ExpandOp::verify() {
  auto sizes = getSizes();
  auto inTy = getInput().getType();

  if (static_cast<size_t>(inTy.getRank()) != sizes.size()) {
    emitError() << "Length of sizes must be equal to rank of input\n";
    return failure();
  }

  for (auto [attr, dim] : llvm::zip(sizes, inTy.getShape())) {
    auto attrVal = attr.cast<IntegerAttr>().getInt();
    if (attrVal == -1 || attrVal == dim) {
      continue;
    }

    if (dim != 1) {
      emitError() << "Dim to be expanded must be 1\n";
      return failure();
    }
  }

  return success();
}

}  // namespace ufront
}  // namespace mlir
