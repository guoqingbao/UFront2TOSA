func.func @forward(%input1: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32>  { 
  %1 = "ufront.conv2d"(%input1) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x3x224x224xf32>) -> tensor<1x24x112x112xf32>
  %2 = "ufront.batchnorm"(%1) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
  %3 = "ufront.relu"(%2) : (tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
  %4 = "ufront.pool2d"(%3) {kernel = [3, 3], pad = [1, 1], pool_type = "POOL_MAX", stride = [2, 2]} : (tensor<1x24x112x112xf32>) -> tensor<1x24x56x56xf32>
  %5 = "ufront.conv2d"(%4) {groups = 24, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x24x56x56xf32>) -> tensor<1x24x28x28xf32>
  %6 = "ufront.batchnorm"(%5) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x24x28x28xf32>) -> tensor<1x24x28x28xf32>
  %7 = "ufront.conv2d"(%6) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x24x28x28xf32>) -> tensor<1x88x28x28xf32>
  %8 = "ufront.batchnorm"(%7) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %9 = "ufront.relu"(%8) : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %10 = "ufront.conv2d"(%4) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x24x56x56xf32>) -> tensor<1x88x56x56xf32>
  %11 = "ufront.batchnorm"(%10) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x56x56xf32>) -> tensor<1x88x56x56xf32>
  %12 = "ufront.relu"(%11) : (tensor<1x88x56x56xf32>) -> tensor<1x88x56x56xf32>
  %13 = "ufront.conv2d"(%12) {groups = 88, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x88x56x56xf32>) -> tensor<1x88x28x28xf32>
  %14 = "ufront.batchnorm"(%13) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %15 = "ufront.conv2d"(%14) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %16 = "ufront.batchnorm"(%15) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %17 = "ufront.relu"(%16) : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %18 = "ufront.concat"(%9, %17) {axis = 1} : (tensor<1x88x28x28xf32>, tensor<1x88x28x28xf32>) -> tensor<1x176x28x28xf32>
  %19 = "ufront.reshape"(%18) {shape = [1, 2, 88, 28, 28]} : (tensor<1x176x28x28xf32>) -> tensor<1x2x88x28x28xf32>
  %20 = "ufront.transpose"(%19) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x88x28x28xf32>) -> tensor<1x88x2x28x28xf32>
  %21 = "ufront.reshape"(%20) {shape = [1, 176, 28, 28]} : (tensor<1x88x2x28x28xf32>) -> tensor<1x176x28x28xf32>
  %22, %23 = "ufront.chunk"(%21) {axis = 1, sizes = 2} : (tensor<1x176x28x28xf32>) -> (tensor<1x88x28x28xf32>, tensor<1x88x28x28xf32>)
  %24 = "ufront.conv2d"(%23) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %25 = "ufront.batchnorm"(%24) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %26 = "ufront.relu"(%25) : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %27 = "ufront.conv2d"(%26) {groups = 88, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %28 = "ufront.batchnorm"(%27) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %29 = "ufront.conv2d"(%28) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %30 = "ufront.batchnorm"(%29) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %31 = "ufront.relu"(%30) : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %32 = "ufront.concat"(%22, %31) {axis = 1} : (tensor<1x88x28x28xf32>, tensor<1x88x28x28xf32>) -> tensor<1x176x28x28xf32>
  %33 = "ufront.reshape"(%32) {shape = [1, 2, 88, 28, 28]} : (tensor<1x176x28x28xf32>) -> tensor<1x2x88x28x28xf32>
  %34 = "ufront.transpose"(%33) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x88x28x28xf32>) -> tensor<1x88x2x28x28xf32>
  %35 = "ufront.reshape"(%34) {shape = [1, 176, 28, 28]} : (tensor<1x88x2x28x28xf32>) -> tensor<1x176x28x28xf32>
  %x, %36 = "ufront.chunk"(%35) {axis = 1, sizes = 2} : (tensor<1x176x28x28xf32>) -> (tensor<1x88x28x28xf32>, tensor<1x88x28x28xf32>)
  %37 = "ufront.conv2d"(%36) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %38 = "ufront.batchnorm"(%37) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %39 = "ufront.relu"(%38) : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %40 = "ufront.conv2d"(%39) {groups = 88, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %41 = "ufront.batchnorm"(%40) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %42 = "ufront.conv2d"(%41) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %43 = "ufront.batchnorm"(%42) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %44 = "ufront.relu"(%43) : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %45 = "ufront.concat"(%x, %44) {axis = 1} : (tensor<1x88x28x28xf32>, tensor<1x88x28x28xf32>) -> tensor<1x176x28x28xf32>
  %46 = "ufront.reshape"(%45) {shape = [1, 2, 88, 28, 28]} : (tensor<1x176x28x28xf32>) -> tensor<1x2x88x28x28xf32>
  %47 = "ufront.transpose"(%46) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x88x28x28xf32>) -> tensor<1x88x2x28x28xf32>
  %48 = "ufront.reshape"(%47) {shape = [1, 176, 28, 28]} : (tensor<1x88x2x28x28xf32>) -> tensor<1x176x28x28xf32>
  %49, %50 = "ufront.chunk"(%48) {axis = 1, sizes = 2} : (tensor<1x176x28x28xf32>) -> (tensor<1x88x28x28xf32>, tensor<1x88x28x28xf32>)
  %51 = "ufront.conv2d"(%50) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %52 = "ufront.batchnorm"(%51) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %53 = "ufront.relu"(%52) : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %54 = "ufront.conv2d"(%53) {groups = 88, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %55 = "ufront.batchnorm"(%54) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %56 = "ufront.conv2d"(%55) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %57 = "ufront.batchnorm"(%56) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %58 = "ufront.relu"(%57) : (tensor<1x88x28x28xf32>) -> tensor<1x88x28x28xf32>
  %59 = "ufront.concat"(%49, %58) {axis = 1} : (tensor<1x88x28x28xf32>, tensor<1x88x28x28xf32>) -> tensor<1x176x28x28xf32>
  %60 = "ufront.reshape"(%59) {shape = [1, 2, 88, 28, 28]} : (tensor<1x176x28x28xf32>) -> tensor<1x2x88x28x28xf32>
  %61 = "ufront.transpose"(%60) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x88x28x28xf32>) -> tensor<1x88x2x28x28xf32>
  %62 = "ufront.reshape"(%61) {shape = [1, 176, 28, 28]} : (tensor<1x88x2x28x28xf32>) -> tensor<1x176x28x28xf32>
  %63 = "ufront.conv2d"(%62) {groups = 176, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x176x28x28xf32>) -> tensor<1x176x14x14xf32>
  %64 = "ufront.batchnorm"(%63) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %65 = "ufront.conv2d"(%64) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %66 = "ufront.batchnorm"(%65) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %67 = "ufront.relu"(%66) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %68 = "ufront.conv2d"(%62) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x28x28xf32>) -> tensor<1x176x28x28xf32>
  %69 = "ufront.batchnorm"(%68) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x28x28xf32>) -> tensor<1x176x28x28xf32>
  %70 = "ufront.relu"(%69) : (tensor<1x176x28x28xf32>) -> tensor<1x176x28x28xf32>
  %71 = "ufront.conv2d"(%70) {groups = 176, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x176x28x28xf32>) -> tensor<1x176x14x14xf32>
  %72 = "ufront.batchnorm"(%71) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %73 = "ufront.conv2d"(%72) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %74 = "ufront.batchnorm"(%73) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %75 = "ufront.relu"(%74) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %76 = "ufront.concat"(%67, %75) {axis = 1} : (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>) -> tensor<1x352x14x14xf32>
  %77 = "ufront.reshape"(%76) {shape = [1, 2, 176, 14, 14]} : (tensor<1x352x14x14xf32>) -> tensor<1x2x176x14x14xf32>
  %78 = "ufront.transpose"(%77) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x176x14x14xf32>) -> tensor<1x176x2x14x14xf32>
  %79 = "ufront.reshape"(%78) {shape = [1, 352, 14, 14]} : (tensor<1x176x2x14x14xf32>) -> tensor<1x352x14x14xf32>
  %80, %81 = "ufront.chunk"(%79) {axis = 1, sizes = 2} : (tensor<1x352x14x14xf32>) -> (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>)
  %82 = "ufront.conv2d"(%81) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %83 = "ufront.batchnorm"(%82) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %84 = "ufront.relu"(%83) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %85 = "ufront.conv2d"(%84) {groups = 176, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %86 = "ufront.batchnorm"(%85) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %87 = "ufront.conv2d"(%86) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %88 = "ufront.batchnorm"(%87) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %89 = "ufront.relu"(%88) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %90 = "ufront.concat"(%80, %89) {axis = 1} : (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>) -> tensor<1x352x14x14xf32>
  %91 = "ufront.reshape"(%90) {shape = [1, 2, 176, 14, 14]} : (tensor<1x352x14x14xf32>) -> tensor<1x2x176x14x14xf32>
  %92 = "ufront.transpose"(%91) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x176x14x14xf32>) -> tensor<1x176x2x14x14xf32>
  %93 = "ufront.reshape"(%92) {shape = [1, 352, 14, 14]} : (tensor<1x176x2x14x14xf32>) -> tensor<1x352x14x14xf32>
  %94, %95 = "ufront.chunk"(%93) {axis = 1, sizes = 2} : (tensor<1x352x14x14xf32>) -> (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>)
  %96 = "ufront.conv2d"(%95) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %97 = "ufront.batchnorm"(%96) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %98 = "ufront.relu"(%97) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %99 = "ufront.conv2d"(%98) {groups = 176, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %100 = "ufront.batchnorm"(%99) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %101 = "ufront.conv2d"(%100) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %102 = "ufront.batchnorm"(%101) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %103 = "ufront.relu"(%102) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %104 = "ufront.concat"(%94, %103) {axis = 1} : (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>) -> tensor<1x352x14x14xf32>
  %105 = "ufront.reshape"(%104) {shape = [1, 2, 176, 14, 14]} : (tensor<1x352x14x14xf32>) -> tensor<1x2x176x14x14xf32>
  %106 = "ufront.transpose"(%105) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x176x14x14xf32>) -> tensor<1x176x2x14x14xf32>
  %107 = "ufront.reshape"(%106) {shape = [1, 352, 14, 14]} : (tensor<1x176x2x14x14xf32>) -> tensor<1x352x14x14xf32>
  %108, %109 = "ufront.chunk"(%107) {axis = 1, sizes = 2} : (tensor<1x352x14x14xf32>) -> (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>)
  %110 = "ufront.conv2d"(%109) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %111 = "ufront.batchnorm"(%110) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %112 = "ufront.relu"(%111) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %113 = "ufront.conv2d"(%112) {groups = 176, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %114 = "ufront.batchnorm"(%113) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %115 = "ufront.conv2d"(%114) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %116 = "ufront.batchnorm"(%115) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %117 = "ufront.relu"(%116) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %118 = "ufront.concat"(%108, %117) {axis = 1} : (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>) -> tensor<1x352x14x14xf32>
  %119 = "ufront.reshape"(%118) {shape = [1, 2, 176, 14, 14]} : (tensor<1x352x14x14xf32>) -> tensor<1x2x176x14x14xf32>
  %120 = "ufront.transpose"(%119) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x176x14x14xf32>) -> tensor<1x176x2x14x14xf32>
  %121 = "ufront.reshape"(%120) {shape = [1, 352, 14, 14]} : (tensor<1x176x2x14x14xf32>) -> tensor<1x352x14x14xf32>
  %122, %123 = "ufront.chunk"(%121) {axis = 1, sizes = 2} : (tensor<1x352x14x14xf32>) -> (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>)
  %124 = "ufront.conv2d"(%123) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %125 = "ufront.batchnorm"(%124) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %126 = "ufront.relu"(%125) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %127 = "ufront.conv2d"(%126) {groups = 176, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %128 = "ufront.batchnorm"(%127) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %129 = "ufront.conv2d"(%128) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %130 = "ufront.batchnorm"(%129) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %131 = "ufront.relu"(%130) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %132 = "ufront.concat"(%122, %131) {axis = 1} : (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>) -> tensor<1x352x14x14xf32>
  %133 = "ufront.reshape"(%132) {shape = [1, 2, 176, 14, 14]} : (tensor<1x352x14x14xf32>) -> tensor<1x2x176x14x14xf32>
  %134 = "ufront.transpose"(%133) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x176x14x14xf32>) -> tensor<1x176x2x14x14xf32>
  %135 = "ufront.reshape"(%134) {shape = [1, 352, 14, 14]} : (tensor<1x176x2x14x14xf32>) -> tensor<1x352x14x14xf32>
  %136, %137 = "ufront.chunk"(%135) {axis = 1, sizes = 2} : (tensor<1x352x14x14xf32>) -> (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>)
  %138 = "ufront.conv2d"(%137) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %139 = "ufront.batchnorm"(%138) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %140 = "ufront.relu"(%139) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %141 = "ufront.conv2d"(%140) {groups = 176, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %142 = "ufront.batchnorm"(%141) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %143 = "ufront.conv2d"(%142) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %144 = "ufront.batchnorm"(%143) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %145 = "ufront.relu"(%144) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %146 = "ufront.concat"(%136, %145) {axis = 1} : (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>) -> tensor<1x352x14x14xf32>
  %147 = "ufront.reshape"(%146) {shape = [1, 2, 176, 14, 14]} : (tensor<1x352x14x14xf32>) -> tensor<1x2x176x14x14xf32>
  %148 = "ufront.transpose"(%147) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x176x14x14xf32>) -> tensor<1x176x2x14x14xf32>
  %149 = "ufront.reshape"(%148) {shape = [1, 352, 14, 14]} : (tensor<1x176x2x14x14xf32>) -> tensor<1x352x14x14xf32>
  %150, %151 = "ufront.chunk"(%149) {axis = 1, sizes = 2} : (tensor<1x352x14x14xf32>) -> (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>)
  %152 = "ufront.conv2d"(%151) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %153 = "ufront.batchnorm"(%152) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %154 = "ufront.relu"(%153) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %155 = "ufront.conv2d"(%154) {groups = 176, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %156 = "ufront.batchnorm"(%155) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %157 = "ufront.conv2d"(%156) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %158 = "ufront.batchnorm"(%157) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %159 = "ufront.relu"(%158) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %160 = "ufront.concat"(%150, %159) {axis = 1} : (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>) -> tensor<1x352x14x14xf32>
  %161 = "ufront.reshape"(%160) {shape = [1, 2, 176, 14, 14]} : (tensor<1x352x14x14xf32>) -> tensor<1x2x176x14x14xf32>
  %162 = "ufront.transpose"(%161) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x176x14x14xf32>) -> tensor<1x176x2x14x14xf32>
  %163 = "ufront.reshape"(%162) {shape = [1, 352, 14, 14]} : (tensor<1x176x2x14x14xf32>) -> tensor<1x352x14x14xf32>
  %164, %165 = "ufront.chunk"(%163) {axis = 1, sizes = 2} : (tensor<1x352x14x14xf32>) -> (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>)
  %166 = "ufront.conv2d"(%165) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %167 = "ufront.batchnorm"(%166) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %168 = "ufront.relu"(%167) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %169 = "ufront.conv2d"(%168) {groups = 176, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %170 = "ufront.batchnorm"(%169) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %171 = "ufront.conv2d"(%170) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %172 = "ufront.batchnorm"(%171) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %173 = "ufront.relu"(%172) : (tensor<1x176x14x14xf32>) -> tensor<1x176x14x14xf32>
  %174 = "ufront.concat"(%164, %173) {axis = 1} : (tensor<1x176x14x14xf32>, tensor<1x176x14x14xf32>) -> tensor<1x352x14x14xf32>
  %175 = "ufront.reshape"(%174) {shape = [1, 2, 176, 14, 14]} : (tensor<1x352x14x14xf32>) -> tensor<1x2x176x14x14xf32>
  %176 = "ufront.transpose"(%175) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x176x14x14xf32>) -> tensor<1x176x2x14x14xf32>
  %177 = "ufront.reshape"(%176) {shape = [1, 352, 14, 14]} : (tensor<1x176x2x14x14xf32>) -> tensor<1x352x14x14xf32>
  %178 = "ufront.conv2d"(%177) {groups = 352, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x352x14x14xf32>) -> tensor<1x352x7x7xf32>
  %179 = "ufront.batchnorm"(%178) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %180 = "ufront.conv2d"(%179) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %181 = "ufront.batchnorm"(%180) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %182 = "ufront.relu"(%181) : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %183 = "ufront.conv2d"(%177) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x352x14x14xf32>) -> tensor<1x352x14x14xf32>
  %184 = "ufront.batchnorm"(%183) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x14x14xf32>) -> tensor<1x352x14x14xf32>
  %185 = "ufront.relu"(%184) : (tensor<1x352x14x14xf32>) -> tensor<1x352x14x14xf32>
  %186 = "ufront.conv2d"(%185) {groups = 352, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x352x14x14xf32>) -> tensor<1x352x7x7xf32>
  %187 = "ufront.batchnorm"(%186) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %188 = "ufront.conv2d"(%187) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %189 = "ufront.batchnorm"(%188) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %190 = "ufront.relu"(%189) : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %191 = "ufront.concat"(%182, %190) {axis = 1} : (tensor<1x352x7x7xf32>, tensor<1x352x7x7xf32>) -> tensor<1x704x7x7xf32>
  %192 = "ufront.reshape"(%191) {shape = [1, 2, 352, 7, 7]} : (tensor<1x704x7x7xf32>) -> tensor<1x2x352x7x7xf32>
  %193 = "ufront.transpose"(%192) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x352x7x7xf32>) -> tensor<1x352x2x7x7xf32>
  %194 = "ufront.reshape"(%193) {shape = [1, 704, 7, 7]} : (tensor<1x352x2x7x7xf32>) -> tensor<1x704x7x7xf32>
  %195, %196 = "ufront.chunk"(%194) {axis = 1, sizes = 2} : (tensor<1x704x7x7xf32>) -> (tensor<1x352x7x7xf32>, tensor<1x352x7x7xf32>)
  %197 = "ufront.conv2d"(%196) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %198 = "ufront.batchnorm"(%197) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %199 = "ufront.relu"(%198) : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %200 = "ufront.conv2d"(%199) {groups = 352, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %201 = "ufront.batchnorm"(%200) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %202 = "ufront.conv2d"(%201) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %203 = "ufront.batchnorm"(%202) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %204 = "ufront.relu"(%203) : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %205 = "ufront.concat"(%195, %204) {axis = 1} : (tensor<1x352x7x7xf32>, tensor<1x352x7x7xf32>) -> tensor<1x704x7x7xf32>
  %206 = "ufront.reshape"(%205) {shape = [1, 2, 352, 7, 7]} : (tensor<1x704x7x7xf32>) -> tensor<1x2x352x7x7xf32>
  %207 = "ufront.transpose"(%206) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x352x7x7xf32>) -> tensor<1x352x2x7x7xf32>
  %208 = "ufront.reshape"(%207) {shape = [1, 704, 7, 7]} : (tensor<1x352x2x7x7xf32>) -> tensor<1x704x7x7xf32>
  %209, %210 = "ufront.chunk"(%208) {axis = 1, sizes = 2} : (tensor<1x704x7x7xf32>) -> (tensor<1x352x7x7xf32>, tensor<1x352x7x7xf32>)
  %211 = "ufront.conv2d"(%210) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %212 = "ufront.batchnorm"(%211) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %213 = "ufront.relu"(%212) : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %214 = "ufront.conv2d"(%213) {groups = 352, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %215 = "ufront.batchnorm"(%214) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %216 = "ufront.conv2d"(%215) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %217 = "ufront.batchnorm"(%216) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %218 = "ufront.relu"(%217) : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %219 = "ufront.concat"(%209, %218) {axis = 1} : (tensor<1x352x7x7xf32>, tensor<1x352x7x7xf32>) -> tensor<1x704x7x7xf32>
  %220 = "ufront.reshape"(%219) {shape = [1, 2, 352, 7, 7]} : (tensor<1x704x7x7xf32>) -> tensor<1x2x352x7x7xf32>
  %221 = "ufront.transpose"(%220) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x352x7x7xf32>) -> tensor<1x352x2x7x7xf32>
  %222 = "ufront.reshape"(%221) {shape = [1, 704, 7, 7]} : (tensor<1x352x2x7x7xf32>) -> tensor<1x704x7x7xf32>
  %223, %224 = "ufront.chunk"(%222) {axis = 1, sizes = 2} : (tensor<1x704x7x7xf32>) -> (tensor<1x352x7x7xf32>, tensor<1x352x7x7xf32>)
  %225 = "ufront.conv2d"(%224) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %226 = "ufront.batchnorm"(%225) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %227 = "ufront.relu"(%226) : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %228 = "ufront.conv2d"(%227) {groups = 352, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %229 = "ufront.batchnorm"(%228) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %230 = "ufront.conv2d"(%229) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %231 = "ufront.batchnorm"(%230) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %232 = "ufront.relu"(%231) : (tensor<1x352x7x7xf32>) -> tensor<1x352x7x7xf32>
  %233 = "ufront.concat"(%223, %232) {axis = 1} : (tensor<1x352x7x7xf32>, tensor<1x352x7x7xf32>) -> tensor<1x704x7x7xf32>
  %234 = "ufront.reshape"(%233) {shape = [1, 2, 352, 7, 7]} : (tensor<1x704x7x7xf32>) -> tensor<1x2x352x7x7xf32>
  %235 = "ufront.transpose"(%234) {perms = [0, 2, 1, 3, 4]} : (tensor<1x2x352x7x7xf32>) -> tensor<1x352x2x7x7xf32>
  %236 = "ufront.reshape"(%235) {shape = [1, 704, 7, 7]} : (tensor<1x352x2x7x7xf32>) -> tensor<1x704x7x7xf32>
  %237 = "ufront.conv2d"(%236) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x704x7x7xf32>) -> tensor<1x1024x7x7xf32>
  %238 = "ufront.batchnorm"(%237) {affine = true, eps = 0.00001, momentum = 0.1, track_running_stats = true} : (tensor<1x1024x7x7xf32>) -> tensor<1x1024x7x7xf32>
  %239 = "ufront.relu"(%238) : (tensor<1x1024x7x7xf32>) -> tensor<1x1024x7x7xf32>
  %240 = "ufront.mean"(%239) {dims = [2, 3], keepdims = false} : (tensor<1x1024x7x7xf32>) -> tensor<1x1024xf32>
  %241 = "ufront.linear"(%240) : (tensor<1x1024xf32>) -> tensor<1x1000xf32>
  return %241: tensor<1x1000xf32>
}