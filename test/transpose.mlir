func.func @transpose (%2 : tensor<1x768x196xf32>) -> tensor<1x196x768xf32> {
  %3 = "ufront.transpose"(%2) {perms = [0, 2, 1]} : (tensor<1x768x196xf32>) -> tensor<1x196x768xf32>
  return %3 : tensor<1x196x768xf32>
}