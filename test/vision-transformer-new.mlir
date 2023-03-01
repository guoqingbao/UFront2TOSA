func.func @forward(%input1: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32>  { 
  %1 = "ufront.conv2d"(%input1) {groups = 1, kernel = [16, 16], pad = [0, 0], stride = [16, 16]} : (tensor<1x3x224x224xf32>) -> tensor<1x768x14x14xf32>
  %2 = "ufront.reshape"(%1) {shape = [1, 768, 196]} : (tensor<1x768x14x14xf32>) -> tensor<1x768x196xf32>
  %3 = "ufront.transpose"(%2) {perm = [0, 2, 1]} : (tensor<1x768x196xf32>) -> tensor<1x196x768xf32>
  %4 = "ufront.parameter"() {requires_grad = true} : () -> tensor<1x1x768xf32>
  %5 = "ufront.expand"(%4) {sizes = [1, -1, -1]} : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
  %6 = "ufront.concat"(%5, %3) {axis = 1} : (tensor<1x1x768xf32>, tensor<1x196x768xf32>) -> tensor<1x197x768xf32>
  %7 = "ufront.parameter"() {requires_grad = true} : () -> tensor<1x197x768xf32>
  %8 = "ufront.add"(%6, %7) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %9 = "ufront.dropout"(%8) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %10 = "ufront.layer_norm"(%9) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %11 = "ufront.multihead_attention"(%10, %10, %10) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %12 = "ufront.dropout"(%11) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %13 = "ufront.add"(%12, %9) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %14 = "ufront.layer_norm"(%13) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %15 = "ufront.linear"(%14) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %16 = "ufront.gelu"(%15) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %17 = "ufront.dropout"(%16) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %18 = "ufront.linear"(%17) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %19 = "ufront.dropout"(%18) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %20 = "ufront.add"(%13, %19) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %21 = "ufront.layer_norm"(%20) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %22 = "ufront.multihead_attention"(%21, %21, %21) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %23 = "ufront.dropout"(%22) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %24 = "ufront.add"(%23, %20) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %25 = "ufront.layer_norm"(%24) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %26 = "ufront.linear"(%25) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %27 = "ufront.gelu"(%26) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %28 = "ufront.dropout"(%27) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %29 = "ufront.linear"(%28) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %30 = "ufront.dropout"(%29) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %31 = "ufront.add"(%24, %30) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %32 = "ufront.layer_norm"(%31) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %33 = "ufront.multihead_attention"(%32, %32, %32) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %34 = "ufront.dropout"(%33) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %35 = "ufront.add"(%34, %31) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %36 = "ufront.layer_norm"(%35) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %37 = "ufront.linear"(%36) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %38 = "ufront.gelu"(%37) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %39 = "ufront.dropout"(%38) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %40 = "ufront.linear"(%39) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %41 = "ufront.dropout"(%40) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %42 = "ufront.add"(%35, %41) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %43 = "ufront.layer_norm"(%42) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %44 = "ufront.multihead_attention"(%43, %43, %43) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %45 = "ufront.dropout"(%44) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %46 = "ufront.add"(%45, %42) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %47 = "ufront.layer_norm"(%46) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %48 = "ufront.linear"(%47) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %49 = "ufront.gelu"(%48) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %50 = "ufront.dropout"(%49) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %51 = "ufront.linear"(%50) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %52 = "ufront.dropout"(%51) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %53 = "ufront.add"(%46, %52) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %54 = "ufront.layer_norm"(%53) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %55 = "ufront.multihead_attention"(%54, %54, %54) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %56 = "ufront.dropout"(%55) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %57 = "ufront.add"(%56, %53) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %58 = "ufront.layer_norm"(%57) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %59 = "ufront.linear"(%58) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %60 = "ufront.gelu"(%59) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %61 = "ufront.dropout"(%60) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %62 = "ufront.linear"(%61) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %63 = "ufront.dropout"(%62) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %64 = "ufront.add"(%57, %63) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %65 = "ufront.layer_norm"(%64) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %66 = "ufront.multihead_attention"(%65, %65, %65) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %67 = "ufront.dropout"(%66) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %68 = "ufront.add"(%67, %64) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %69 = "ufront.layer_norm"(%68) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %70 = "ufront.linear"(%69) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %71 = "ufront.gelu"(%70) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %72 = "ufront.dropout"(%71) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %73 = "ufront.linear"(%72) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %74 = "ufront.dropout"(%73) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %75 = "ufront.add"(%68, %74) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %76 = "ufront.layer_norm"(%75) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %77 = "ufront.multihead_attention"(%76, %76, %76) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %78 = "ufront.dropout"(%77) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %79 = "ufront.add"(%78, %75) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %80 = "ufront.layer_norm"(%79) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %81 = "ufront.linear"(%80) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %82 = "ufront.gelu"(%81) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %83 = "ufront.dropout"(%82) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %84 = "ufront.linear"(%83) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %85 = "ufront.dropout"(%84) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %86 = "ufront.add"(%79, %85) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %87 = "ufront.layer_norm"(%86) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %88 = "ufront.multihead_attention"(%87, %87, %87) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %89 = "ufront.dropout"(%88) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %90 = "ufront.add"(%89, %86) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %91 = "ufront.layer_norm"(%90) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %92 = "ufront.linear"(%91) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %93 = "ufront.gelu"(%92) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %94 = "ufront.dropout"(%93) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %95 = "ufront.linear"(%94) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %96 = "ufront.dropout"(%95) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %97 = "ufront.add"(%90, %96) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %98 = "ufront.layer_norm"(%97) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %99 = "ufront.multihead_attention"(%98, %98, %98) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %100 = "ufront.dropout"(%99) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %101 = "ufront.add"(%100, %97) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %102 = "ufront.layer_norm"(%101) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %103 = "ufront.linear"(%102) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %104 = "ufront.gelu"(%103) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %105 = "ufront.dropout"(%104) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %106 = "ufront.linear"(%105) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %107 = "ufront.dropout"(%106) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %108 = "ufront.add"(%101, %107) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %109 = "ufront.layer_norm"(%108) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %110 = "ufront.multihead_attention"(%109, %109, %109) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %111 = "ufront.dropout"(%110) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %112 = "ufront.add"(%111, %108) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %113 = "ufront.layer_norm"(%112) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %114 = "ufront.linear"(%113) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %115 = "ufront.gelu"(%114) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %116 = "ufront.dropout"(%115) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %117 = "ufront.linear"(%116) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %118 = "ufront.dropout"(%117) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %119 = "ufront.add"(%112, %118) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %120 = "ufront.layer_norm"(%119) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %121 = "ufront.multihead_attention"(%120, %120, %120) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %122 = "ufront.dropout"(%121) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %123 = "ufront.add"(%122, %119) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %124 = "ufront.layer_norm"(%123) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %125 = "ufront.linear"(%124) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %126 = "ufront.gelu"(%125) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %127 = "ufront.dropout"(%126) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %128 = "ufront.linear"(%127) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %129 = "ufront.dropout"(%128) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %130 = "ufront.add"(%123, %129) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %131 = "ufront.layer_norm"(%130) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %132 = "ufront.multihead_attention"(%131, %131, %131) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %133 = "ufront.dropout"(%132) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %134 = "ufront.add"(%133, %130) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %135 = "ufront.layer_norm"(%134) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %136 = "ufront.linear"(%135) : (tensor<1x197x768xf32>) -> tensor<1x197x3072xf32>
  %137 = "ufront.gelu"(%136) : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %138 = "ufront.dropout"(%137) {rate = 0.0, seed = 0} : (tensor<1x197x3072xf32>) -> tensor<1x197x3072xf32>
  %139 = "ufront.linear"(%138) : (tensor<1x197x3072xf32>) -> tensor<1x197x768xf32>
  %140 = "ufront.dropout"(%139) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %141 = "ufront.add"(%134, %140) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %142 = "ufront.layer_norm"(%141) {elementwise_affine = true, eps = 0.000001, normalized_shape = [768]} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
  %143 = "ufront.slice"(%142) {output_shape = [1, 768], slices = [["none", "none", "none"], 0]} : (tensor<1x197x768xf32>) -> tensor<1x768xf32>
  %144 = "ufront.linear"(%143) : (tensor<1x768xf32>) -> tensor<1x1000xf32>
  %145 = "ufront.softmax"(%144) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
  return %145: tensor<1x1000xf32>
}