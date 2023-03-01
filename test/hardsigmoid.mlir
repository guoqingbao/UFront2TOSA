func.func @hardsigmoid(%10 : tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>{
  %11 = "ufront.hardsigmoid"(%10) : (tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
  return %11 : tensor<1x16x1x1xf32>
}