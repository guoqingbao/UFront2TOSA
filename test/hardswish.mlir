func.func @hardswish(%2 : tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32> {
  %3 = "ufront.hardswish"(%2) : (tensor<1x16x112x112xf32>) -> tensor<1x16x112x112xf32>
  return %3 : tensor<1x16x112x112xf32>
}