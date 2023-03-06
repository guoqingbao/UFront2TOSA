func.func @dropout(%6 : tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
  %7 = "ufront.dropout"(%6) {rate = 0.2, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  return %7 : tensor<1x197x768xf32>
}