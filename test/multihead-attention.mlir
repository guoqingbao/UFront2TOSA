func.func @multihead_attention(%8 : tensor<1x197x768xf32>) -> tensor<1x197x768xf32> {
  %9 = "ufront.multihead_attention"(%8, %8, %8) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  return %9 : tensor<1x197x768xf32>
}