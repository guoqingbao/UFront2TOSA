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

}  // namespace ufront
}  // namespace mlir
