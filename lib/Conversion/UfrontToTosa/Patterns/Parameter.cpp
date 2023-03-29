#include <fstream>

#include "../Patterns.hpp"
#include "cnpy.h"

namespace mlir {
namespace ufront {

std::string convertHexToBin(std::string hex) {
  std::string bin;
  for (size_t i = 0; i < hex.length(); i += 2) {
    std::string byte = hex.substr(i, 2);
    char chr = (char)(int)strtol(byte.c_str(), NULL, 16);
    bin.push_back(chr);
  }
  return bin;
}

LogicalResult ParameterConverter::matchAndRewrite(
    ParameterOp param, PatternRewriter& rewriter) const {
  auto initializer = param->getAttrOfType<StringAttr>("initializer");
  if (!initializer) {
    return success();
  }

  auto hex = initializer.str();
  std::string filename = "/tmp/ufronttmp.npz";
  std::ofstream file{filename, std::ios::binary};

  file << convertHexToBin(hex);
  file.close();

  auto dtype = param->getAttrOfType<StringAttr>("dtype");
  if (dtype.str() == "Float") {
    auto load = cnpy::npz_load(filename);
    auto array = load.begin()->second.as_vec<float>();

    SmallVector<APFloat> values;
    transform(array, std::back_inserter(values),
              [](float f) { return APFloat{f}; });

    auto attr = DenseElementsAttr::get(param.getType(), values);
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(param, param.getType(), attr);
  }

  std::remove(filename.c_str());
  return success();
}

}  // namespace ufront
}  // namespace mlir
