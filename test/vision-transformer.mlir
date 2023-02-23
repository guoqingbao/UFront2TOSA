func.func @forward(%input1 : tensor<1x3x224x224xf32>, %class_token : tensor<1x1x768xf32>, %encoder.pos_embedding : tensor<1x197x768xf32>) -> tensor<1x1000xf32>  { 
	%1 = "ufront.conv2d"(%input1) {groups = 1, kernel = [16, 16], padding = [0, 0], stride = [16, 16]} : (tensor<1x3x224x224xf32>) -> tensor<1x768x14x14xf32>
	%2 = "ufront.reshape"(%1) {shape = [1, 768, 196]} : (tensor<1x768x14x14xf32>) -> tensor<1x768x196xf32>
	%3 = "ufront.transpose"(%2) {perms = [0, 2, 1]} : (tensor<1x768x196xf32>) -> tensor<1x196x768xf32>
	%4 = "ufront.expand"(%class_token) {sizes = [1, -1, -1]} : (tensor<1x1x768xf32>) -> tensor<1x1x768xf32>
	%5 = "ufront.concat"(%4, %3) {axis = 1} : (tensor<1x1x768xf32>, tensor<1x196x768xf32>) -> tensor<1x197x768xf32>
	%6 = "ufront.add"(%5, %encoder.pos_embedding) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%7 = "ufront.dropout"(%6) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%8 = "ufront.layer_norm"(%7) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%9 = "ufront.multihead_attention"(%8, %8, %8) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%10 = "ufront.dropout"(%9) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%11 = "ufront.add"(%10, %7) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%12 = "ufront.layer_norm"(%11) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%13 = "ufront.linear"(%12) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%14 = "ufront.gelu"(%13) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%15 = "ufront.dropout"(%14) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%16 = "ufront.linear"(%15) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%17 = "ufront.dropout"(%16) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%18 = "ufront.add"(%11, %17) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%19 = "ufront.layer_norm"(%18) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%20 = "ufront.multihead_attention"(%19, %19, %19) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%21 = "ufront.dropout"(%20) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%22 = "ufront.add"(%21, %18) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%23 = "ufront.layer_norm"(%22) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%24 = "ufront.linear"(%23) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%25 = "ufront.gelu"(%24) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%26 = "ufront.dropout"(%25) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%27 = "ufront.linear"(%26) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%28 = "ufront.dropout"(%27) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%29 = "ufront.add"(%22, %28) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%30 = "ufront.layer_norm"(%29) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%31 = "ufront.multihead_attention"(%30, %30, %30) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%32 = "ufront.dropout"(%31) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%33 = "ufront.add"(%32, %29) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%34 = "ufront.layer_norm"(%33) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%35 = "ufront.linear"(%34) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%36 = "ufront.gelu"(%35) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%37 = "ufront.dropout"(%36) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%38 = "ufront.linear"(%37) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%39 = "ufront.dropout"(%38) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%40 = "ufront.add"(%33, %39) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%41 = "ufront.layer_norm"(%40) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%42 = "ufront.multihead_attention"(%41, %41, %41) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%43 = "ufront.dropout"(%42) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%44 = "ufront.add"(%43, %40) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%45 = "ufront.layer_norm"(%44) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%46 = "ufront.linear"(%45) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%47 = "ufront.gelu"(%46) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%48 = "ufront.dropout"(%47) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%49 = "ufront.linear"(%48) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%50 = "ufront.dropout"(%49) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%51 = "ufront.add"(%44, %50) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%52 = "ufront.layer_norm"(%51) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%53 = "ufront.multihead_attention"(%52, %52, %52) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%54 = "ufront.dropout"(%53) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%55 = "ufront.add"(%54, %51) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%56 = "ufront.layer_norm"(%55) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%57 = "ufront.linear"(%56) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%58 = "ufront.gelu"(%57) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%59 = "ufront.dropout"(%58) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%60 = "ufront.linear"(%59) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%61 = "ufront.dropout"(%60) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%62 = "ufront.add"(%55, %61) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%63 = "ufront.layer_norm"(%62) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%64 = "ufront.multihead_attention"(%63, %63, %63) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%65 = "ufront.dropout"(%64) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%66 = "ufront.add"(%65, %62) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%67 = "ufront.layer_norm"(%66) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%68 = "ufront.linear"(%67) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%69 = "ufront.gelu"(%68) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%70 = "ufront.dropout"(%69) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%71 = "ufront.linear"(%70) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%72 = "ufront.dropout"(%71) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%73 = "ufront.add"(%66, %72) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%74 = "ufront.layer_norm"(%73) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%75 = "ufront.multihead_attention"(%74, %74, %74) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%76 = "ufront.dropout"(%75) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%77 = "ufront.add"(%76, %73) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%78 = "ufront.layer_norm"(%77) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%79 = "ufront.linear"(%78) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%80 = "ufront.gelu"(%79) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%81 = "ufront.dropout"(%80) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%82 = "ufront.linear"(%81) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%83 = "ufront.dropout"(%82) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%84 = "ufront.add"(%77, %83) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%85 = "ufront.layer_norm"(%84) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%86 = "ufront.multihead_attention"(%85, %85, %85) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%87 = "ufront.dropout"(%86) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%88 = "ufront.add"(%87, %84) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%89 = "ufront.layer_norm"(%88) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%90 = "ufront.linear"(%89) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%91 = "ufront.gelu"(%90) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%92 = "ufront.dropout"(%91) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%93 = "ufront.linear"(%92) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%94 = "ufront.dropout"(%93) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%95 = "ufront.add"(%88, %94) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%96 = "ufront.layer_norm"(%95) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%97 = "ufront.multihead_attention"(%96, %96, %96) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%98 = "ufront.dropout"(%97) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%99 = "ufront.add"(%98, %95) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%100 = "ufront.layer_norm"(%99) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%101 = "ufront.linear"(%100) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%102 = "ufront.gelu"(%101) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%103 = "ufront.dropout"(%102) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%104 = "ufront.linear"(%103) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%105 = "ufront.dropout"(%104) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%106 = "ufront.add"(%99, %105) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%107 = "ufront.layer_norm"(%106) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%108 = "ufront.multihead_attention"(%107, %107, %107) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%109 = "ufront.dropout"(%108) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%110 = "ufront.add"(%109, %106) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%111 = "ufront.layer_norm"(%110) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%112 = "ufront.linear"(%111) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%113 = "ufront.gelu"(%112) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%114 = "ufront.dropout"(%113) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%115 = "ufront.linear"(%114) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%116 = "ufront.dropout"(%115) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%117 = "ufront.add"(%110, %116) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%118 = "ufront.layer_norm"(%117) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%119 = "ufront.multihead_attention"(%118, %118, %118) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%120 = "ufront.dropout"(%119) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%121 = "ufront.add"(%120, %117) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%122 = "ufront.layer_norm"(%121) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%123 = "ufront.linear"(%122) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%124 = "ufront.gelu"(%123) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%125 = "ufront.dropout"(%124) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%126 = "ufront.linear"(%125) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%127 = "ufront.dropout"(%126) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%128 = "ufront.add"(%121, %127) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%129 = "ufront.layer_norm"(%128) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%130 = "ufront.multihead_attention"(%129, %129, %129) {batch_first = true, dropout = 0.0, embed_dim = 768, num_heads = 12} : (tensor<1x197x768xf32>, tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%131 = "ufront.dropout"(%130) {rate = 0.0, seed = 0} : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%132 = "ufront.add"(%131, %128) : (tensor<1x197x768xf32>, tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%133 = "ufront.layer_norm"(%132) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%134 = "ufront.linear"(%133) : (tensor<1x197x768xf32>) -> tensor<1x3072xf32>
	%135 = "ufront.gelu"(%134) : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%136 = "ufront.dropout"(%135) {rate = 0.0, seed = 0} : (tensor<1x3072xf32>) -> tensor<1x3072xf32>
	%137 = "ufront.linear"(%136) : (tensor<1x3072xf32>) -> tensor<1x768xf32>
	%138 = "ufront.dropout"(%137) {rate = 0.0, seed = 0} : (tensor<1x768xf32>) -> tensor<1x768xf32>
	%139 = "ufront.add"(%132, %138) : (tensor<1x197x768xf32>, tensor<1x768xf32>) -> tensor<1x197x768xf32>
	%140 = "ufront.layer_norm"(%139) : (tensor<1x197x768xf32>) -> tensor<1x197x768xf32>
	%141 = "ufront.slice"(%140) {output_shape = [1, 768], slices = "(slice(None, None, None), 0)"} : (tensor<1x197x768xf32>) -> tensor<1x768xf32>
	%142 = "ufront.linear"(%141) : (tensor<1x768xf32>) -> tensor<1x1000xf32>
	%143 = "ufront.softmax"(%142) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
	return %143 : tensor<1x1000xf32>
}