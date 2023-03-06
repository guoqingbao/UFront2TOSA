func.func @batch_matmul(%22 : tensor<16x512x512xf32>, %15 : tensor<16x512x8xf32>) -> tensor<16x512x8xf32> {
  %23 = "ufront.batch_matmul"(%22, %15) : (tensor<16x512x512xf32>, tensor<16x512x8xf32>) -> tensor<16x512x8xf32>
  return %23 : tensor<16x512x8xf32>
}