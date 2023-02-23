func.func @expand(%class_token : tensor<1x1x768xf32>) -> tensor<1x1x768xf32> {
  %4 = "ufront.expand"(%class_token) {sizes = [1, -1, -1]} : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
  return %4 : tensor<1x1x768xf32>
}