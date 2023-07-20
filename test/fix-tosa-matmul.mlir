// ufront-opt %s --fix-tosa-matmul | FileCheck %s

// CHECK: @matmul(%[[LHS:.*]]: tensor<2x197x768xf32>, %[[RHS:.*]]: tensor<1x768x768xf32>)
func.func @matmul(%lhs : tensor<2x197x768xf32>, %rhs : tensor<1x768x768xf32>) -> tensor<2x197x768xf32> {
  // CHECK: "tosa.reshape"(%[[RHS]])
  // CHECK: %[[BROADCASTED:.*]] = linalg.broadcast
  // CHECK: %[[RES:.*]] = "tosa.matmul"(%[[LHS]], %[[BROADCASTED]])
  // CHECK: return %[[RES]]
  %res = "tosa.matmul"(%lhs, %rhs) : (tensor<2x197x768xf32>, tensor<1x768x768xf32>) -> tensor<2x197x768xf32>
  return %res : tensor<2x197x768xf32>
}
