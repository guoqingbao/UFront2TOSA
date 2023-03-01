func.func @forward(%input1 :  tensor<1x3x224x224xf32>) -> tensor<1x1000xf32>  { 
	%1 = "ufront.conv2d"(%input1) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x3x224x224xf32>) -> tensor<1x24x112x112xf32>
	%2 = "ufront.batchnorm"(%1) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
	%3 = "ufront.silu"(%2) : (tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
	%4 = "ufront.conv2d"(%3) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
	%5 = "ufront.batchnorm"(%4) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
	%6 = "ufront.silu"(%5) : (tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
	%7 = "ufront.add"(%6, %3) : (tensor<1x24x112x112xf32>, tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
	%8 = "ufront.conv2d"(%7) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
	%9 = "ufront.batchnorm"(%8) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
	%10 = "ufront.silu"(%9) : (tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
	%11 = "ufront.add"(%10, %7) : (tensor<1x24x112x112xf32>, tensor<1x24x112x112xf32>) -> tensor<1x24x112x112xf32>
	%12 = "ufront.conv2d"(%11) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x24x112x112xf32>) -> tensor<1x96x56x56xf32>
	%13 = "ufront.batchnorm"(%12) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
	%14 = "ufront.silu"(%13) : (tensor<1x96x56x56xf32>) -> tensor<1x96x56x56xf32>
	%15 = "ufront.conv2d"(%14) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x96x56x56xf32>) -> tensor<1x48x56x56xf32>
	%16 = "ufront.batchnorm"(%15) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x48x56x56xf32>) -> tensor<1x48x56x56xf32>
	%17 = "ufront.conv2d"(%16) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x48x56x56xf32>) -> tensor<1x192x56x56xf32>
	%18 = "ufront.batchnorm"(%17) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x192x56x56xf32>) -> tensor<1x192x56x56xf32>
	%19 = "ufront.silu"(%18) : (tensor<1x192x56x56xf32>) -> tensor<1x192x56x56xf32>
	%20 = "ufront.conv2d"(%19) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x192x56x56xf32>) -> tensor<1x48x56x56xf32>
	%21 = "ufront.batchnorm"(%20) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x48x56x56xf32>) -> tensor<1x48x56x56xf32>
	%22 = "ufront.add"(%21, %16) : (tensor<1x48x56x56xf32>, tensor<1x48x56x56xf32>) -> tensor<1x48x56x56xf32>
	%23 = "ufront.conv2d"(%22) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x48x56x56xf32>) -> tensor<1x192x56x56xf32>
	%24 = "ufront.batchnorm"(%23) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x192x56x56xf32>) -> tensor<1x192x56x56xf32>
	%25 = "ufront.silu"(%24) : (tensor<1x192x56x56xf32>) -> tensor<1x192x56x56xf32>
	%26 = "ufront.conv2d"(%25) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x192x56x56xf32>) -> tensor<1x48x56x56xf32>
	%27 = "ufront.batchnorm"(%26) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x48x56x56xf32>) -> tensor<1x48x56x56xf32>
	%28 = "ufront.add"(%27, %22) : (tensor<1x48x56x56xf32>, tensor<1x48x56x56xf32>) -> tensor<1x48x56x56xf32>
	%29 = "ufront.conv2d"(%28) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x48x56x56xf32>) -> tensor<1x192x56x56xf32>
	%30 = "ufront.batchnorm"(%29) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x192x56x56xf32>) -> tensor<1x192x56x56xf32>
	%31 = "ufront.silu"(%30) : (tensor<1x192x56x56xf32>) -> tensor<1x192x56x56xf32>
	%32 = "ufront.conv2d"(%31) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x192x56x56xf32>) -> tensor<1x48x56x56xf32>
	%33 = "ufront.batchnorm"(%32) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x48x56x56xf32>) -> tensor<1x48x56x56xf32>
	%34 = "ufront.add"(%33, %28) : (tensor<1x48x56x56xf32>, tensor<1x48x56x56xf32>) -> tensor<1x48x56x56xf32>
	%35 = "ufront.conv2d"(%34) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x48x56x56xf32>) -> tensor<1x192x28x28xf32>
	%36 = "ufront.batchnorm"(%35) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
	%37 = "ufront.silu"(%36) : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
	%38 = "ufront.conv2d"(%37) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x192x28x28xf32>) -> tensor<1x64x28x28xf32>
	%39 = "ufront.batchnorm"(%38) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
	%40 = "ufront.conv2d"(%39) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x64x28x28xf32>) -> tensor<1x256x28x28xf32>
	%41 = "ufront.batchnorm"(%40) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
	%42 = "ufront.silu"(%41) : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
	%43 = "ufront.conv2d"(%42) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x28x28xf32>) -> tensor<1x64x28x28xf32>
	%44 = "ufront.batchnorm"(%43) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
	%45 = "ufront.add"(%44, %39) : (tensor<1x64x28x28xf32>, tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
	%46 = "ufront.conv2d"(%45) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x64x28x28xf32>) -> tensor<1x256x28x28xf32>
	%47 = "ufront.batchnorm"(%46) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
	%48 = "ufront.silu"(%47) : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
	%49 = "ufront.conv2d"(%48) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x28x28xf32>) -> tensor<1x64x28x28xf32>
	%50 = "ufront.batchnorm"(%49) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
	%51 = "ufront.add"(%50, %45) : (tensor<1x64x28x28xf32>, tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
	%52 = "ufront.conv2d"(%51) {groups = 1, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x64x28x28xf32>) -> tensor<1x256x28x28xf32>
	%53 = "ufront.batchnorm"(%52) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
	%54 = "ufront.silu"(%53) : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
	%55 = "ufront.conv2d"(%54) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x28x28xf32>) -> tensor<1x64x28x28xf32>
	%56 = "ufront.batchnorm"(%55) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
	%57 = "ufront.add"(%56, %51) : (tensor<1x64x28x28xf32>, tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
	%58 = "ufront.conv2d"(%57) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x28x28xf32>) -> tensor<1x256x28x28xf32>
	%59 = "ufront.batchnorm"(%58) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
	%60 = "ufront.silu"(%59) : (tensor<1x256x28x28xf32>) -> tensor<1x256x28x28xf32>
	%61 = "ufront.conv2d"(%60) {groups = 256, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x256x28x28xf32>) -> tensor<1x256x14x14xf32>
	%62 = "ufront.batchnorm"(%61) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
	%63 = "ufront.silu"(%62) : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
	%64 = "ufront.pool2d"(%63) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x256x14x14xf32>) -> tensor<1x256x1x1xf32>
	%65 = "ufront.conv2d"(%64) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x1x1xf32>) -> tensor<1x16x1x1xf32>
	%66 = "ufront.silu"(%65) : (tensor<1x16x1x1xf32>) -> tensor<1x16x1x1xf32>
	%67 = "ufront.conv2d"(%66) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x16x1x1xf32>) -> tensor<1x256x1x1xf32>
	%68 = "ufront.sigmoid"(%67) : (tensor<1x256x1x1xf32>) -> tensor<1x256x1x1xf32>
	%69 = "ufront.multiply"(%68, %63) : (tensor<1x256x1x1xf32>, tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
	%70 = "ufront.conv2d"(%69) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x14x14xf32>) -> tensor<1x128x14x14xf32>
	%71 = "ufront.batchnorm"(%70) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
	%72 = "ufront.conv2d"(%71) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x128x14x14xf32>) -> tensor<1x512x14x14xf32>
	%73 = "ufront.batchnorm"(%72) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%74 = "ufront.silu"(%73) : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%75 = "ufront.conv2d"(%74) {groups = 512, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%76 = "ufront.batchnorm"(%75) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%77 = "ufront.silu"(%76) : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%78 = "ufront.pool2d"(%77) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x512x14x14xf32>) -> tensor<1x512x1x1xf32>
	%79 = "ufront.conv2d"(%78) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x512x1x1xf32>) -> tensor<1x32x1x1xf32>
	%80 = "ufront.silu"(%79) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
	%81 = "ufront.conv2d"(%80) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x32x1x1xf32>) -> tensor<1x512x1x1xf32>
	%82 = "ufront.sigmoid"(%81) : (tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32>
	%83 = "ufront.multiply"(%82, %77) : (tensor<1x512x1x1xf32>, tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%84 = "ufront.conv2d"(%83) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x512x14x14xf32>) -> tensor<1x128x14x14xf32>
	%85 = "ufront.batchnorm"(%84) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
	%86 = "ufront.add"(%85, %71) : (tensor<1x128x14x14xf32>, tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
	%87 = "ufront.conv2d"(%86) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x128x14x14xf32>) -> tensor<1x512x14x14xf32>
	%88 = "ufront.batchnorm"(%87) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%89 = "ufront.silu"(%88) : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%90 = "ufront.conv2d"(%89) {groups = 512, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%91 = "ufront.batchnorm"(%90) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%92 = "ufront.silu"(%91) : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%93 = "ufront.pool2d"(%92) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x512x14x14xf32>) -> tensor<1x512x1x1xf32>
	%94 = "ufront.conv2d"(%93) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x512x1x1xf32>) -> tensor<1x32x1x1xf32>
	%95 = "ufront.silu"(%94) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
	%96 = "ufront.conv2d"(%95) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x32x1x1xf32>) -> tensor<1x512x1x1xf32>
	%97 = "ufront.sigmoid"(%96) : (tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32>
	%98 = "ufront.multiply"(%97, %92) : (tensor<1x512x1x1xf32>, tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%99 = "ufront.conv2d"(%98) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x512x14x14xf32>) -> tensor<1x128x14x14xf32>
	%100 = "ufront.batchnorm"(%99) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
	%101 = "ufront.add"(%100, %86) : (tensor<1x128x14x14xf32>, tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
	%102 = "ufront.conv2d"(%101) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x128x14x14xf32>) -> tensor<1x512x14x14xf32>
	%103 = "ufront.batchnorm"(%102) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%104 = "ufront.silu"(%103) : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%105 = "ufront.conv2d"(%104) {groups = 512, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%106 = "ufront.batchnorm"(%105) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%107 = "ufront.silu"(%106) : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%108 = "ufront.pool2d"(%107) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x512x14x14xf32>) -> tensor<1x512x1x1xf32>
	%109 = "ufront.conv2d"(%108) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x512x1x1xf32>) -> tensor<1x32x1x1xf32>
	%110 = "ufront.silu"(%109) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
	%111 = "ufront.conv2d"(%110) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x32x1x1xf32>) -> tensor<1x512x1x1xf32>
	%112 = "ufront.sigmoid"(%111) : (tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32>
	%113 = "ufront.multiply"(%112, %107) : (tensor<1x512x1x1xf32>, tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%114 = "ufront.conv2d"(%113) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x512x14x14xf32>) -> tensor<1x128x14x14xf32>
	%115 = "ufront.batchnorm"(%114) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
	%116 = "ufront.add"(%115, %101) : (tensor<1x128x14x14xf32>, tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
	%117 = "ufront.conv2d"(%116) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x128x14x14xf32>) -> tensor<1x512x14x14xf32>
	%118 = "ufront.batchnorm"(%117) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%119 = "ufront.silu"(%118) : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%120 = "ufront.conv2d"(%119) {groups = 512, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%121 = "ufront.batchnorm"(%120) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%122 = "ufront.silu"(%121) : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%123 = "ufront.pool2d"(%122) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x512x14x14xf32>) -> tensor<1x512x1x1xf32>
	%124 = "ufront.conv2d"(%123) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x512x1x1xf32>) -> tensor<1x32x1x1xf32>
	%125 = "ufront.silu"(%124) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
	%126 = "ufront.conv2d"(%125) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x32x1x1xf32>) -> tensor<1x512x1x1xf32>
	%127 = "ufront.sigmoid"(%126) : (tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32>
	%128 = "ufront.multiply"(%127, %122) : (tensor<1x512x1x1xf32>, tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%129 = "ufront.conv2d"(%128) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x512x14x14xf32>) -> tensor<1x128x14x14xf32>
	%130 = "ufront.batchnorm"(%129) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
	%131 = "ufront.add"(%130, %116) : (tensor<1x128x14x14xf32>, tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
	%132 = "ufront.conv2d"(%131) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x128x14x14xf32>) -> tensor<1x512x14x14xf32>
	%133 = "ufront.batchnorm"(%132) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%134 = "ufront.silu"(%133) : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%135 = "ufront.conv2d"(%134) {groups = 512, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%136 = "ufront.batchnorm"(%135) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%137 = "ufront.silu"(%136) : (tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%138 = "ufront.pool2d"(%137) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x512x14x14xf32>) -> tensor<1x512x1x1xf32>
	%139 = "ufront.conv2d"(%138) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x512x1x1xf32>) -> tensor<1x32x1x1xf32>
	%140 = "ufront.silu"(%139) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
	%141 = "ufront.conv2d"(%140) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x32x1x1xf32>) -> tensor<1x512x1x1xf32>
	%142 = "ufront.sigmoid"(%141) : (tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32>
	%143 = "ufront.multiply"(%142, %137) : (tensor<1x512x1x1xf32>, tensor<1x512x14x14xf32>) -> tensor<1x512x14x14xf32>
	%144 = "ufront.conv2d"(%143) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x512x14x14xf32>) -> tensor<1x128x14x14xf32>
	%145 = "ufront.batchnorm"(%144) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
	%146 = "ufront.add"(%145, %131) : (tensor<1x128x14x14xf32>, tensor<1x128x14x14xf32>) -> tensor<1x128x14x14xf32>
	%147 = "ufront.conv2d"(%146) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x128x14x14xf32>) -> tensor<1x768x14x14xf32>
	%148 = "ufront.batchnorm"(%147) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x768x14x14xf32>) -> tensor<1x768x14x14xf32>
	%149 = "ufront.silu"(%148) : (tensor<1x768x14x14xf32>) -> tensor<1x768x14x14xf32>
	%150 = "ufront.conv2d"(%149) {groups = 768, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x768x14x14xf32>) -> tensor<1x768x14x14xf32>
	%151 = "ufront.batchnorm"(%150) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x768x14x14xf32>) -> tensor<1x768x14x14xf32>
	%152 = "ufront.silu"(%151) : (tensor<1x768x14x14xf32>) -> tensor<1x768x14x14xf32>
	%153 = "ufront.pool2d"(%152) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x768x14x14xf32>) -> tensor<1x768x1x1xf32>
	%154 = "ufront.conv2d"(%153) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x768x1x1xf32>) -> tensor<1x32x1x1xf32>
	%155 = "ufront.silu"(%154) : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
	%156 = "ufront.conv2d"(%155) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x32x1x1xf32>) -> tensor<1x768x1x1xf32>
	%157 = "ufront.sigmoid"(%156) : (tensor<1x768x1x1xf32>) -> tensor<1x768x1x1xf32>
	%158 = "ufront.multiply"(%157, %152) : (tensor<1x768x1x1xf32>, tensor<1x768x14x14xf32>) -> tensor<1x768x14x14xf32>
	%159 = "ufront.conv2d"(%158) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x768x14x14xf32>) -> tensor<1x160x14x14xf32>
	%160 = "ufront.batchnorm"(%159) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%161 = "ufront.conv2d"(%160) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x160x14x14xf32>) -> tensor<1x960x14x14xf32>
	%162 = "ufront.batchnorm"(%161) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%163 = "ufront.silu"(%162) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%164 = "ufront.conv2d"(%163) {groups = 960, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%165 = "ufront.batchnorm"(%164) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%166 = "ufront.silu"(%165) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%167 = "ufront.pool2d"(%166) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x960x14x14xf32>) -> tensor<1x960x1x1xf32>
	%168 = "ufront.conv2d"(%167) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x1x1xf32>) -> tensor<1x40x1x1xf32>
	%169 = "ufront.silu"(%168) : (tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
	%170 = "ufront.conv2d"(%169) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x40x1x1xf32>) -> tensor<1x960x1x1xf32>
	%171 = "ufront.sigmoid"(%170) : (tensor<1x960x1x1xf32>) -> tensor<1x960x1x1xf32>
	%172 = "ufront.multiply"(%171, %166) : (tensor<1x960x1x1xf32>, tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%173 = "ufront.conv2d"(%172) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x160x14x14xf32>
	%174 = "ufront.batchnorm"(%173) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%175 = "ufront.add"(%174, %160) : (tensor<1x160x14x14xf32>, tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%176 = "ufront.conv2d"(%175) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x160x14x14xf32>) -> tensor<1x960x14x14xf32>
	%177 = "ufront.batchnorm"(%176) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%178 = "ufront.silu"(%177) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%179 = "ufront.conv2d"(%178) {groups = 960, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%180 = "ufront.batchnorm"(%179) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%181 = "ufront.silu"(%180) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%182 = "ufront.pool2d"(%181) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x960x14x14xf32>) -> tensor<1x960x1x1xf32>
	%183 = "ufront.conv2d"(%182) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x1x1xf32>) -> tensor<1x40x1x1xf32>
	%184 = "ufront.silu"(%183) : (tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
	%185 = "ufront.conv2d"(%184) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x40x1x1xf32>) -> tensor<1x960x1x1xf32>
	%186 = "ufront.sigmoid"(%185) : (tensor<1x960x1x1xf32>) -> tensor<1x960x1x1xf32>
	%187 = "ufront.multiply"(%186, %181) : (tensor<1x960x1x1xf32>, tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%188 = "ufront.conv2d"(%187) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x160x14x14xf32>
	%189 = "ufront.batchnorm"(%188) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%190 = "ufront.add"(%189, %175) : (tensor<1x160x14x14xf32>, tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%191 = "ufront.conv2d"(%190) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x160x14x14xf32>) -> tensor<1x960x14x14xf32>
	%192 = "ufront.batchnorm"(%191) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%193 = "ufront.silu"(%192) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%194 = "ufront.conv2d"(%193) {groups = 960, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%195 = "ufront.batchnorm"(%194) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%196 = "ufront.silu"(%195) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%197 = "ufront.pool2d"(%196) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x960x14x14xf32>) -> tensor<1x960x1x1xf32>
	%198 = "ufront.conv2d"(%197) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x1x1xf32>) -> tensor<1x40x1x1xf32>
	%199 = "ufront.silu"(%198) : (tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
	%200 = "ufront.conv2d"(%199) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x40x1x1xf32>) -> tensor<1x960x1x1xf32>
	%201 = "ufront.sigmoid"(%200) : (tensor<1x960x1x1xf32>) -> tensor<1x960x1x1xf32>
	%202 = "ufront.multiply"(%201, %196) : (tensor<1x960x1x1xf32>, tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%203 = "ufront.conv2d"(%202) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x160x14x14xf32>
	%204 = "ufront.batchnorm"(%203) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%205 = "ufront.add"(%204, %190) : (tensor<1x160x14x14xf32>, tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%206 = "ufront.conv2d"(%205) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x160x14x14xf32>) -> tensor<1x960x14x14xf32>
	%207 = "ufront.batchnorm"(%206) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%208 = "ufront.silu"(%207) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%209 = "ufront.conv2d"(%208) {groups = 960, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%210 = "ufront.batchnorm"(%209) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%211 = "ufront.silu"(%210) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%212 = "ufront.pool2d"(%211) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x960x14x14xf32>) -> tensor<1x960x1x1xf32>
	%213 = "ufront.conv2d"(%212) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x1x1xf32>) -> tensor<1x40x1x1xf32>
	%214 = "ufront.silu"(%213) : (tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
	%215 = "ufront.conv2d"(%214) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x40x1x1xf32>) -> tensor<1x960x1x1xf32>
	%216 = "ufront.sigmoid"(%215) : (tensor<1x960x1x1xf32>) -> tensor<1x960x1x1xf32>
	%217 = "ufront.multiply"(%216, %211) : (tensor<1x960x1x1xf32>, tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%218 = "ufront.conv2d"(%217) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x160x14x14xf32>
	%219 = "ufront.batchnorm"(%218) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%220 = "ufront.add"(%219, %205) : (tensor<1x160x14x14xf32>, tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%221 = "ufront.conv2d"(%220) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x160x14x14xf32>) -> tensor<1x960x14x14xf32>
	%222 = "ufront.batchnorm"(%221) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%223 = "ufront.silu"(%222) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%224 = "ufront.conv2d"(%223) {groups = 960, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%225 = "ufront.batchnorm"(%224) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%226 = "ufront.silu"(%225) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%227 = "ufront.pool2d"(%226) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x960x14x14xf32>) -> tensor<1x960x1x1xf32>
	%228 = "ufront.conv2d"(%227) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x1x1xf32>) -> tensor<1x40x1x1xf32>
	%229 = "ufront.silu"(%228) : (tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
	%230 = "ufront.conv2d"(%229) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x40x1x1xf32>) -> tensor<1x960x1x1xf32>
	%231 = "ufront.sigmoid"(%230) : (tensor<1x960x1x1xf32>) -> tensor<1x960x1x1xf32>
	%232 = "ufront.multiply"(%231, %226) : (tensor<1x960x1x1xf32>, tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%233 = "ufront.conv2d"(%232) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x160x14x14xf32>
	%234 = "ufront.batchnorm"(%233) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%235 = "ufront.add"(%234, %220) : (tensor<1x160x14x14xf32>, tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%236 = "ufront.conv2d"(%235) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x160x14x14xf32>) -> tensor<1x960x14x14xf32>
	%237 = "ufront.batchnorm"(%236) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%238 = "ufront.silu"(%237) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%239 = "ufront.conv2d"(%238) {groups = 960, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%240 = "ufront.batchnorm"(%239) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%241 = "ufront.silu"(%240) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%242 = "ufront.pool2d"(%241) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x960x14x14xf32>) -> tensor<1x960x1x1xf32>
	%243 = "ufront.conv2d"(%242) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x1x1xf32>) -> tensor<1x40x1x1xf32>
	%244 = "ufront.silu"(%243) : (tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
	%245 = "ufront.conv2d"(%244) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x40x1x1xf32>) -> tensor<1x960x1x1xf32>
	%246 = "ufront.sigmoid"(%245) : (tensor<1x960x1x1xf32>) -> tensor<1x960x1x1xf32>
	%247 = "ufront.multiply"(%246, %241) : (tensor<1x960x1x1xf32>, tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%248 = "ufront.conv2d"(%247) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x160x14x14xf32>
	%249 = "ufront.batchnorm"(%248) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%250 = "ufront.add"(%249, %235) : (tensor<1x160x14x14xf32>, tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%251 = "ufront.conv2d"(%250) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x160x14x14xf32>) -> tensor<1x960x14x14xf32>
	%252 = "ufront.batchnorm"(%251) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%253 = "ufront.silu"(%252) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%254 = "ufront.conv2d"(%253) {groups = 960, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%255 = "ufront.batchnorm"(%254) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%256 = "ufront.silu"(%255) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%257 = "ufront.pool2d"(%256) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x960x14x14xf32>) -> tensor<1x960x1x1xf32>
	%258 = "ufront.conv2d"(%257) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x1x1xf32>) -> tensor<1x40x1x1xf32>
	%259 = "ufront.silu"(%258) : (tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
	%260 = "ufront.conv2d"(%259) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x40x1x1xf32>) -> tensor<1x960x1x1xf32>
	%261 = "ufront.sigmoid"(%260) : (tensor<1x960x1x1xf32>) -> tensor<1x960x1x1xf32>
	%262 = "ufront.multiply"(%261, %256) : (tensor<1x960x1x1xf32>, tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%263 = "ufront.conv2d"(%262) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x160x14x14xf32>
	%264 = "ufront.batchnorm"(%263) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%265 = "ufront.add"(%264, %250) : (tensor<1x160x14x14xf32>, tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%266 = "ufront.conv2d"(%265) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x160x14x14xf32>) -> tensor<1x960x14x14xf32>
	%267 = "ufront.batchnorm"(%266) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%268 = "ufront.silu"(%267) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%269 = "ufront.conv2d"(%268) {groups = 960, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%270 = "ufront.batchnorm"(%269) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%271 = "ufront.silu"(%270) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%272 = "ufront.pool2d"(%271) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x960x14x14xf32>) -> tensor<1x960x1x1xf32>
	%273 = "ufront.conv2d"(%272) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x1x1xf32>) -> tensor<1x40x1x1xf32>
	%274 = "ufront.silu"(%273) : (tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
	%275 = "ufront.conv2d"(%274) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x40x1x1xf32>) -> tensor<1x960x1x1xf32>
	%276 = "ufront.sigmoid"(%275) : (tensor<1x960x1x1xf32>) -> tensor<1x960x1x1xf32>
	%277 = "ufront.multiply"(%276, %271) : (tensor<1x960x1x1xf32>, tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%278 = "ufront.conv2d"(%277) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x14x14xf32>) -> tensor<1x160x14x14xf32>
	%279 = "ufront.batchnorm"(%278) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%280 = "ufront.add"(%279, %265) : (tensor<1x160x14x14xf32>, tensor<1x160x14x14xf32>) -> tensor<1x160x14x14xf32>
	%281 = "ufront.conv2d"(%280) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x160x14x14xf32>) -> tensor<1x960x14x14xf32>
	%282 = "ufront.batchnorm"(%281) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%283 = "ufront.silu"(%282) : (tensor<1x960x14x14xf32>) -> tensor<1x960x14x14xf32>
	%284 = "ufront.conv2d"(%283) {groups = 960, kernel = [3, 3], pad = [1, 1], stride = [2, 2]} : (tensor<1x960x14x14xf32>) -> tensor<1x960x7x7xf32>
	%285 = "ufront.batchnorm"(%284) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
	%286 = "ufront.silu"(%285) : (tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
	%287 = "ufront.pool2d"(%286) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x960x7x7xf32>) -> tensor<1x960x1x1xf32>
	%288 = "ufront.conv2d"(%287) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x1x1xf32>) -> tensor<1x40x1x1xf32>
	%289 = "ufront.silu"(%288) : (tensor<1x40x1x1xf32>) -> tensor<1x40x1x1xf32>
	%290 = "ufront.conv2d"(%289) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x40x1x1xf32>) -> tensor<1x960x1x1xf32>
	%291 = "ufront.sigmoid"(%290) : (tensor<1x960x1x1xf32>) -> tensor<1x960x1x1xf32>
	%292 = "ufront.multiply"(%291, %286) : (tensor<1x960x1x1xf32>, tensor<1x960x7x7xf32>) -> tensor<1x960x7x7xf32>
	%293 = "ufront.conv2d"(%292) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x960x7x7xf32>) -> tensor<1x256x7x7xf32>
	%294 = "ufront.batchnorm"(%293) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%295 = "ufront.conv2d"(%294) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%296 = "ufront.batchnorm"(%295) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%297 = "ufront.silu"(%296) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%298 = "ufront.conv2d"(%297) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%299 = "ufront.batchnorm"(%298) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%300 = "ufront.silu"(%299) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%301 = "ufront.pool2d"(%300) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%302 = "ufront.conv2d"(%301) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%303 = "ufront.silu"(%302) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%304 = "ufront.conv2d"(%303) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%305 = "ufront.sigmoid"(%304) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%306 = "ufront.multiply"(%305, %300) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%307 = "ufront.conv2d"(%306) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%308 = "ufront.batchnorm"(%307) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%309 = "ufront.add"(%308, %294) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%310 = "ufront.conv2d"(%309) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%311 = "ufront.batchnorm"(%310) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%312 = "ufront.silu"(%311) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%313 = "ufront.conv2d"(%312) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%314 = "ufront.batchnorm"(%313) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%315 = "ufront.silu"(%314) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%316 = "ufront.pool2d"(%315) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%317 = "ufront.conv2d"(%316) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%318 = "ufront.silu"(%317) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%319 = "ufront.conv2d"(%318) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%320 = "ufront.sigmoid"(%319) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%321 = "ufront.multiply"(%320, %315) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%322 = "ufront.conv2d"(%321) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%323 = "ufront.batchnorm"(%322) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%324 = "ufront.add"(%323, %309) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%325 = "ufront.conv2d"(%324) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%326 = "ufront.batchnorm"(%325) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%327 = "ufront.silu"(%326) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%328 = "ufront.conv2d"(%327) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%329 = "ufront.batchnorm"(%328) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%330 = "ufront.silu"(%329) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%331 = "ufront.pool2d"(%330) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%332 = "ufront.conv2d"(%331) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%333 = "ufront.silu"(%332) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%334 = "ufront.conv2d"(%333) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%335 = "ufront.sigmoid"(%334) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%336 = "ufront.multiply"(%335, %330) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%337 = "ufront.conv2d"(%336) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%338 = "ufront.batchnorm"(%337) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%339 = "ufront.add"(%338, %324) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%340 = "ufront.conv2d"(%339) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%341 = "ufront.batchnorm"(%340) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%342 = "ufront.silu"(%341) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%343 = "ufront.conv2d"(%342) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%344 = "ufront.batchnorm"(%343) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%345 = "ufront.silu"(%344) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%346 = "ufront.pool2d"(%345) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%347 = "ufront.conv2d"(%346) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%348 = "ufront.silu"(%347) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%349 = "ufront.conv2d"(%348) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%350 = "ufront.sigmoid"(%349) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%351 = "ufront.multiply"(%350, %345) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%352 = "ufront.conv2d"(%351) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%353 = "ufront.batchnorm"(%352) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%354 = "ufront.add"(%353, %339) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%355 = "ufront.conv2d"(%354) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%356 = "ufront.batchnorm"(%355) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%357 = "ufront.silu"(%356) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%358 = "ufront.conv2d"(%357) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%359 = "ufront.batchnorm"(%358) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%360 = "ufront.silu"(%359) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%361 = "ufront.pool2d"(%360) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%362 = "ufront.conv2d"(%361) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%363 = "ufront.silu"(%362) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%364 = "ufront.conv2d"(%363) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%365 = "ufront.sigmoid"(%364) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%366 = "ufront.multiply"(%365, %360) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%367 = "ufront.conv2d"(%366) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%368 = "ufront.batchnorm"(%367) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%369 = "ufront.add"(%368, %354) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%370 = "ufront.conv2d"(%369) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%371 = "ufront.batchnorm"(%370) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%372 = "ufront.silu"(%371) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%373 = "ufront.conv2d"(%372) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%374 = "ufront.batchnorm"(%373) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%375 = "ufront.silu"(%374) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%376 = "ufront.pool2d"(%375) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%377 = "ufront.conv2d"(%376) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%378 = "ufront.silu"(%377) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%379 = "ufront.conv2d"(%378) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%380 = "ufront.sigmoid"(%379) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%381 = "ufront.multiply"(%380, %375) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%382 = "ufront.conv2d"(%381) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%383 = "ufront.batchnorm"(%382) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%384 = "ufront.add"(%383, %369) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%385 = "ufront.conv2d"(%384) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%386 = "ufront.batchnorm"(%385) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%387 = "ufront.silu"(%386) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%388 = "ufront.conv2d"(%387) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%389 = "ufront.batchnorm"(%388) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%390 = "ufront.silu"(%389) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%391 = "ufront.pool2d"(%390) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%392 = "ufront.conv2d"(%391) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%393 = "ufront.silu"(%392) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%394 = "ufront.conv2d"(%393) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%395 = "ufront.sigmoid"(%394) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%396 = "ufront.multiply"(%395, %390) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%397 = "ufront.conv2d"(%396) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%398 = "ufront.batchnorm"(%397) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%399 = "ufront.add"(%398, %384) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%400 = "ufront.conv2d"(%399) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%401 = "ufront.batchnorm"(%400) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%402 = "ufront.silu"(%401) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%403 = "ufront.conv2d"(%402) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%404 = "ufront.batchnorm"(%403) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%405 = "ufront.silu"(%404) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%406 = "ufront.pool2d"(%405) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%407 = "ufront.conv2d"(%406) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%408 = "ufront.silu"(%407) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%409 = "ufront.conv2d"(%408) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%410 = "ufront.sigmoid"(%409) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%411 = "ufront.multiply"(%410, %405) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%412 = "ufront.conv2d"(%411) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%413 = "ufront.batchnorm"(%412) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%414 = "ufront.add"(%413, %399) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%415 = "ufront.conv2d"(%414) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%416 = "ufront.batchnorm"(%415) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%417 = "ufront.silu"(%416) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%418 = "ufront.conv2d"(%417) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%419 = "ufront.batchnorm"(%418) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%420 = "ufront.silu"(%419) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%421 = "ufront.pool2d"(%420) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%422 = "ufront.conv2d"(%421) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%423 = "ufront.silu"(%422) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%424 = "ufront.conv2d"(%423) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%425 = "ufront.sigmoid"(%424) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%426 = "ufront.multiply"(%425, %420) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%427 = "ufront.conv2d"(%426) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%428 = "ufront.batchnorm"(%427) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%429 = "ufront.add"(%428, %414) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%430 = "ufront.conv2d"(%429) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%431 = "ufront.batchnorm"(%430) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%432 = "ufront.silu"(%431) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%433 = "ufront.conv2d"(%432) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%434 = "ufront.batchnorm"(%433) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%435 = "ufront.silu"(%434) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%436 = "ufront.pool2d"(%435) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%437 = "ufront.conv2d"(%436) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%438 = "ufront.silu"(%437) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%439 = "ufront.conv2d"(%438) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%440 = "ufront.sigmoid"(%439) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%441 = "ufront.multiply"(%440, %435) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%442 = "ufront.conv2d"(%441) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%443 = "ufront.batchnorm"(%442) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%444 = "ufront.add"(%443, %429) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%445 = "ufront.conv2d"(%444) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%446 = "ufront.batchnorm"(%445) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%447 = "ufront.silu"(%446) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%448 = "ufront.conv2d"(%447) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%449 = "ufront.batchnorm"(%448) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%450 = "ufront.silu"(%449) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%451 = "ufront.pool2d"(%450) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%452 = "ufront.conv2d"(%451) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%453 = "ufront.silu"(%452) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%454 = "ufront.conv2d"(%453) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%455 = "ufront.sigmoid"(%454) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%456 = "ufront.multiply"(%455, %450) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%457 = "ufront.conv2d"(%456) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%458 = "ufront.batchnorm"(%457) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%459 = "ufront.add"(%458, %444) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%460 = "ufront.conv2d"(%459) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%461 = "ufront.batchnorm"(%460) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%462 = "ufront.silu"(%461) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%463 = "ufront.conv2d"(%462) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%464 = "ufront.batchnorm"(%463) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%465 = "ufront.silu"(%464) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%466 = "ufront.pool2d"(%465) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%467 = "ufront.conv2d"(%466) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%468 = "ufront.silu"(%467) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%469 = "ufront.conv2d"(%468) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%470 = "ufront.sigmoid"(%469) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%471 = "ufront.multiply"(%470, %465) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%472 = "ufront.conv2d"(%471) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%473 = "ufront.batchnorm"(%472) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%474 = "ufront.add"(%473, %459) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%475 = "ufront.conv2d"(%474) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%476 = "ufront.batchnorm"(%475) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%477 = "ufront.silu"(%476) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%478 = "ufront.conv2d"(%477) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%479 = "ufront.batchnorm"(%478) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%480 = "ufront.silu"(%479) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%481 = "ufront.pool2d"(%480) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%482 = "ufront.conv2d"(%481) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%483 = "ufront.silu"(%482) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%484 = "ufront.conv2d"(%483) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%485 = "ufront.sigmoid"(%484) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%486 = "ufront.multiply"(%485, %480) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%487 = "ufront.conv2d"(%486) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%488 = "ufront.batchnorm"(%487) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%489 = "ufront.add"(%488, %474) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%490 = "ufront.conv2d"(%489) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%491 = "ufront.batchnorm"(%490) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%492 = "ufront.silu"(%491) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%493 = "ufront.conv2d"(%492) {groups = 1536, kernel = [3, 3], pad = [1, 1], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%494 = "ufront.batchnorm"(%493) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%495 = "ufront.silu"(%494) : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%496 = "ufront.pool2d"(%495) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1536x7x7xf32>) -> tensor<1x1536x1x1xf32>
	%497 = "ufront.conv2d"(%496) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x1x1xf32>) -> tensor<1x64x1x1xf32>
	%498 = "ufront.silu"(%497) : (tensor<1x64x1x1xf32>) -> tensor<1x64x1x1xf32>
	%499 = "ufront.conv2d"(%498) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x64x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%500 = "ufront.sigmoid"(%499) : (tensor<1x1536x1x1xf32>) -> tensor<1x1536x1x1xf32>
	%501 = "ufront.multiply"(%500, %495) : (tensor<1x1536x1x1xf32>, tensor<1x1536x7x7xf32>) -> tensor<1x1536x7x7xf32>
	%502 = "ufront.conv2d"(%501) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x1536x7x7xf32>) -> tensor<1x256x7x7xf32>
	%503 = "ufront.batchnorm"(%502) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%504 = "ufront.add"(%503, %489) : (tensor<1x256x7x7xf32>, tensor<1x256x7x7xf32>) -> tensor<1x256x7x7xf32>
	%505 = "ufront.conv2d"(%504) {groups = 1, kernel = [1, 1], pad = [0, 0], stride = [1, 1]} : (tensor<1x256x7x7xf32>) -> tensor<1x1280x7x7xf32>
	%506 = "ufront.batchnorm"(%505) {affine = true, eps = 0.001, momentum = 0.1, track_running_stats = true} : (tensor<1x1280x7x7xf32>) -> tensor<1x1280x7x7xf32>
	%507 = "ufront.silu"(%506) : (tensor<1x1280x7x7xf32>) -> tensor<1x1280x7x7xf32>
	%508 = "ufront.pool2d"(%507) {output_size = [1, 1], pool_type = "POOL_ADAPTIVE"} : (tensor<1x1280x7x7xf32>) -> tensor<1x1280x1x1xf32>
	%509 = "ufront.flat"(%508) {end_dim = -1, start_dim = 1} : (tensor<1x1280x1x1xf32>) -> tensor<1x1280xf32>
	%510 = "ufront.dropout"(%509) {rate = 0.2, seed = 0} : (tensor<1x1280xf32>) -> tensor<1x1280xf32>
	%511 = "ufront.linear"(%510) : (tensor<1x1280xf32>) -> tensor<1x1000xf32>
	%512 = "ufront.softmax"(%511) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
	return %512 :  tensor<1x1000xf32>
}