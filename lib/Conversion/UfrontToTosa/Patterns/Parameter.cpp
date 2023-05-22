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

float* char_to_pointer(std::string input) {
    return (float*)std::stoul(input, nullptr, 16);
}

LogicalResult ParameterConverter::matchAndRewrite(
    ParameterOp param, PatternRewriter& rewriter) const {
  auto initializer = param->getAttrOfType<StringAttr>("initializer");
  if (!initializer) {
    return success();
  }

  auto hex = initializer.str();
  if (hex.size() < 20 && hex[0]=='0' and hex[1]=='x')
  {
    auto dtype = param->getAttrOfType<StringAttr>("dtype");
    if (dtype.str() == "Float") {
      auto output = param.getTensor();
      auto outTy = output.getType();
      auto outShape = outTy.getShape();
      auto outsize = [](ArrayRef<int64_t> shape) {
        int ret = 1;
        for (auto size : shape) {
          ret *= size;
        }
        return ret;
      };
      int size = outsize(outShape);
      // printf("size %d, %s, %d, %d, %d, %d", size, hex.c_str(), outShape[0], outShape[1], outShape[2], outShape[3]);
      //TODO hex to memory address, obtain vector from memory
      float* array = char_to_pointer(hex);
      // printf("C++ address %s, 100th value %f\n", hex.c_str(), array[100]);
      std::vector<float> vec {array, array + size};

      SmallVector<APFloat> values;
      transform(vec, std::back_inserter(values),
                [](float f) { return APFloat{f}; });

      auto attr = DenseElementsAttr::get(param.getType(), values);
      rewriter.replaceOpWithNewOp<tosa::ConstOp>(param, param.getType(), attr);
    }

  } else {
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
  }

  return success();
}

}  // namespace ufront
}  // namespace mlir
