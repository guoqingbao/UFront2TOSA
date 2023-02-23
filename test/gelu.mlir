func.func @gelu(%13 : tensor<1x3072xf32>) -> tensor<1x3072xf32> {
  %14 = "ufront.gelu"(%13) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
  return %14 : tensor<1x3072xf32>
}