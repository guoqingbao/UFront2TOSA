#ifndef TEST_PARAMETER_TO_CONST_HPP
#define TEST_PARAMETER_TO_CONST_HPP

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ufront {

#define GEN_PASS_DEF_TESTCONVERTPARAMETERTOCONST
#define GEN_PASS_DECL_TESTCONVERTPARAMETERTOCONST
#include "Conversion/Passes.hpp.inc"

std::unique_ptr<Pass> createTestConvertParameterToConst();

}  // namespace ufront
}  // namespace mlir

#endif  // TEST_PARAMETER_TO_CONST_HPP
