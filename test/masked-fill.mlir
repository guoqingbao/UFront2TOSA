func.func @masked_fill(%19 : tensor<16x512x512xf32>, %20 : tensor<16x512x512xf32>) -> tensor<16x512x512xf32> {
  %21 = "ufront.masked_fill"(%19, %20) {value = -1000000000.0} : (tensor<16x512x512xf32>, tensor<16x512x512xf32>) -> tensor<16x512x512xf32>
  return %21 : tensor<16x512x512xf32>
}