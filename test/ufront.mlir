func.func @forward(%input1: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32>  { 
  %1 = "ufront.conv2d"(%input1) {kernel = [7, 7], groups = 1, stride = [2, 2], padding = [3, 3]} : (tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32>
  %2 = "ufront.batchnorm"(%1) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
  %3 = "ufront.relu"(%2) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
  %4 = "ufront.pool2d"(%3) {stride = [2, 2], kernel = [3, 3], padding = [1, 1], pool_type = "PoolType.POOL_MAX"} : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
  %5 = "ufront.conv2d"(%4) {kernel = [3, 3], groups = 1, stride = [1, 1], padding = [1, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %6 = "ufront.batchnorm"(%5) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %7 = "ufront.relu"(%6) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %8 = "ufront.conv2d"(%7) {padding = [1, 1], stride = [1, 1], groups = 1, kernel = [3, 3]} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %9 = "ufront.batchnorm"(%8) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %10 = "ufront.add"(%9, %4) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %11 = "ufront.relu"(%10) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %12 = "ufront.conv2d"(%11) {groups = 1, padding = [1, 1], stride = [1, 1], kernel = [3, 3]} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %13 = "ufront.batchnorm"(%12) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %14 = "ufront.relu"(%13) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %15 = "ufront.conv2d"(%14) {padding = [1, 1], groups = 1, stride = [1, 1], kernel = [3, 3]} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %16 = "ufront.batchnorm"(%15) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %17 = "ufront.add"(%16, %11) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %18 = "ufront.relu"(%17) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  %19 = "ufront.conv2d"(%18) {stride = [2, 2], padding = [1, 1], kernel = [3, 3], groups = 1} : (tensor<1x64x56x56xf32>) -> tensor<1x128x28x28xf32>
  %20 = "ufront.batchnorm"(%19) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %21 = "ufront.relu"(%20) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %22 = "ufront.conv2d"(%21) {groups = 1, stride = [1, 1], kernel = [3, 3], padding = [1, 1]} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %23 = "ufront.batchnorm"(%22) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %24 = "ufront.conv2d"(%18) {groups = 1, padding = [0, 0], stride = [2, 2], kernel = [1, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x128x28x28xf32>
  %25 = "ufront.batchnorm"(%24) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %26 = "ufront.add"(%23, %25) : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %27 = "ufront.relu"(%26) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %28 = "ufront.conv2d"(%27) {groups = 1, padding = [1, 1], kernel = [3, 3], stride = [1, 1]} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %29 = "ufront.batchnorm"(%28) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %30 = "ufront.relu"(%29) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %31 = "ufront.conv2d"(%30) {groups = 1, kernel = [3, 3], stride = [1, 1], padding = [1, 1]} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %32 = "ufront.batchnorm"(%31) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %33 = "ufront.add"(%32, %27) : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %34 = "ufront.relu"(%33) : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
  %35 = "ufront.conv2d"(%34) {stride = [2, 2], padding = [1, 1], groups = 1, kernel = [3, 3]} : (tensor<1x128x28x28xf32>) -> tensor<1x256x14x14xf32>
  %36 = "ufront.batchnorm"(%35) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %37 = "ufront.relu"(%36) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %38 = "ufront.conv2d"(%37) {padding = [1, 1], kernel = [3, 3], stride = [1, 1], groups = 1} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %39 = "ufront.batchnorm"(%38) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %40 = "ufront.conv2d"(%34) {kernel = [1, 1], groups = 1, stride = [2, 2], padding = [0, 0]} : (tensor<1x128x28x28xf32>) -> tensor<1x256x14x14xf32>
  %41 = "ufront.batchnorm"(%40) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %42 = "ufront.add"(%39, %41) : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %43 = "ufront.relu"(%42) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %44 = "ufront.conv2d"(%43) {groups = 1, kernel = [3, 3], stride = [1, 1], padding = [1, 1]} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %45 = "ufront.batchnorm"(%44) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %46 = "ufront.relu"(%45) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %47 = "ufront.conv2d"(%46) {kernel = [3, 3], stride = [1, 1], groups = 1, padding = [1, 1]} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %48 = "ufront.batchnorm"(%47) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %49 = "ufront.add"(%48, %43) : (tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %50 = "ufront.relu"(%49) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
  %51 = "ufront.conv2d"(%50) {kernel = [3, 3], groups = 1, stride = [2, 2], padding = [1, 1]} : (tensor<1x256x14x14xf32>) -> tensor<1x512x7x7xf32>
  %52 = "ufront.batchnorm"(%51) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %53 = "ufront.relu"(%52) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %54 = "ufront.conv2d"(%53) {groups = 1, stride = [1, 1], kernel = [3, 3], padding = [1, 1]} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %55 = "ufront.batchnorm"(%54) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %56 = "ufront.conv2d"(%50) {stride = [2, 2], padding = [0, 0], kernel = [1, 1], groups = 1} : (tensor<1x256x14x14xf32>) -> tensor<1x512x7x7xf32>
  %57 = "ufront.batchnorm"(%56) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %58 = "ufront.add"(%55, %57) : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %59 = "ufront.relu"(%58) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %60 = "ufront.conv2d"(%59) {padding = [1, 1], stride = [1, 1], groups = 1, kernel = [3, 3]} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %61 = "ufront.batchnorm"(%60) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %62 = "ufront.relu"(%61) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %63 = "ufront.conv2d"(%62) {kernel = [3, 3], stride = [1, 1], padding = [1, 1], groups = 1} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %64 = "ufront.batchnorm"(%63) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %65 = "ufront.add"(%64, %59) : (tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %66 = "ufront.relu"(%65) : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
  %67 = "ufront.pool2d"(%66) {pool_type = "PoolType.POOL_ADAPTIVE", output_size = [1, 1]} : (tensor<1x512x7x7xf32>) -> tensor<1x512x1x1xf32>
  %68 = "ufront.flat"(%67) : (tensor<1x512x1x1xf32>) -> tensor<1x512xf32>
  %69 = "ufront.linear"(%68) : (tensor<1x512xf32>) -> tensor<1x1000xf32>
  %70 = "ufront.softmax"(%69) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
  return %70: tensor<1x1000xf32>
}