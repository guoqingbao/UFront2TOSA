func.func @forward(%input1: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32>  { 
  %1="ufront.conv2d"(%input1){groups=1, kernel=[3, 3], pad=[0, 0], stride=[2, 2]}:(tensor<1x3x224x224xf32>) -> tensor<1x32x111x111xf32>
  %2="ufront.batchnorm"(%1){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x32x111x111xf32>) -> tensor<1x32x111x111xf32>
  %3="ufront.relu"(%2):(tensor<1x32x111x111xf32>) -> tensor<1x32x111x111xf32>
  %4="ufront.conv2d"(%3){groups=1, kernel=[3, 3], pad=[0, 0], stride=[1, 1]}:(tensor<1x32x111x111xf32>) -> tensor<1x32x109x109xf32>
  %5="ufront.batchnorm"(%4){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x32x109x109xf32>) -> tensor<1x32x109x109xf32>
  %6="ufront.relu"(%5):(tensor<1x32x109x109xf32>) -> tensor<1x32x109x109xf32>
  %7="ufront.conv2d"(%6){groups=1, kernel=[3, 3], pad=[1, 1], stride=[1, 1]}:(tensor<1x32x109x109xf32>) -> tensor<1x64x109x109xf32>
  %8="ufront.batchnorm"(%7){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x109x109xf32>) -> tensor<1x64x109x109xf32>
  %9="ufront.relu"(%8):(tensor<1x64x109x109xf32>) -> tensor<1x64x109x109xf32>
  %10="ufront.pool2d"(%9){kernel=[3, 3], pad=[0, 0], pool_type="POOL_MAX", stride=[2, 2]}:(tensor<1x64x109x109xf32>) -> tensor<1x64x54x54xf32>
  %11="ufront.conv2d"(%10){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x64x54x54xf32>) -> tensor<1x80x54x54xf32>
  %12="ufront.batchnorm"(%11){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x80x54x54xf32>) -> tensor<1x80x54x54xf32>
  %13="ufront.relu"(%12):(tensor<1x80x54x54xf32>) -> tensor<1x80x54x54xf32>
  %14="ufront.conv2d"(%13){groups=1, kernel=[3, 3], pad=[0, 0], stride=[1, 1]}:(tensor<1x80x54x54xf32>) -> tensor<1x192x52x52xf32>
  %15="ufront.batchnorm"(%14){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x52x52xf32>) -> tensor<1x192x52x52xf32>
  %16="ufront.relu"(%15):(tensor<1x192x52x52xf32>) -> tensor<1x192x52x52xf32>
  %17="ufront.pool2d"(%16){kernel=[3, 3], pad=[0, 0], pool_type="POOL_MAX", stride=[2, 2]}:(tensor<1x192x52x52xf32>) -> tensor<1x192x26x26xf32>
  %18="ufront.conv2d"(%17){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x192x26x26xf32>) -> tensor<1x64x26x26xf32>
  %19="ufront.batchnorm"(%18){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %20="ufront.relu"(%19):(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %21="ufront.conv2d"(%17){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x192x26x26xf32>) -> tensor<1x48x26x26xf32>
  %22="ufront.batchnorm"(%21){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x48x26x26xf32>) -> tensor<1x48x26x26xf32>
  %23="ufront.relu"(%22):(tensor<1x48x26x26xf32>) -> tensor<1x48x26x26xf32>
  %24="ufront.conv2d"(%23){groups=1, kernel=[5, 5], pad=[2, 2], stride=[1, 1]}:(tensor<1x48x26x26xf32>) -> tensor<1x64x26x26xf32>
  %25="ufront.batchnorm"(%24){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %26="ufront.relu"(%25):(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %27="ufront.conv2d"(%17){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x192x26x26xf32>) -> tensor<1x64x26x26xf32>
  %28="ufront.batchnorm"(%27){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %29="ufront.relu"(%28):(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %30="ufront.conv2d"(%29){groups=1, kernel=[3, 3], pad=[1, 1], stride=[1, 1]}:(tensor<1x64x26x26xf32>) -> tensor<1x96x26x26xf32>
  %31="ufront.batchnorm"(%30){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %32="ufront.relu"(%31):(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %33="ufront.conv2d"(%32){groups=1, kernel=[3, 3], pad=[1, 1], stride=[1, 1]}:(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %34="ufront.batchnorm"(%33){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %35="ufront.relu"(%34):(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %36="ufront.pool2d"(%17){kernel=[3, 3], pad=[1, 1], pool_type="POOL_AVG", stride=[1, 1]}:(tensor<1x192x26x26xf32>) -> tensor<1x192x13x13xf32>
  %37="ufront.conv2d"(%36){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x192x13x13xf32>) -> tensor<1x32x13x13xf32>
  %38="ufront.batchnorm"(%37){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x32x13x13xf32>) -> tensor<1x32x13x13xf32>
  %39="ufront.relu"(%38):(tensor<1x32x13x13xf32>) -> tensor<1x32x13x13xf32>
  %40="ufront.concat"(%20, %26, %35, %39){axis=1}:(tensor<1x64x26x26xf32>, tensor<1x64x26x26xf32>, tensor<1x96x26x26xf32>, tensor<1x32x13x13xf32>) -> tensor<1x256x26x26xf32>
  %41="ufront.conv2d"(%40){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x256x26x26xf32>) -> tensor<1x64x26x26xf32>
  %42="ufront.batchnorm"(%41){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %43="ufront.relu"(%42):(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %44="ufront.conv2d"(%40){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x256x26x26xf32>) -> tensor<1x48x26x26xf32>
  %45="ufront.batchnorm"(%44){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x48x26x26xf32>) -> tensor<1x48x26x26xf32>
  %46="ufront.relu"(%45):(tensor<1x48x26x26xf32>) -> tensor<1x48x26x26xf32>
  %47="ufront.conv2d"(%46){groups=1, kernel=[5, 5], pad=[2, 2], stride=[1, 1]}:(tensor<1x48x26x26xf32>) -> tensor<1x64x26x26xf32>
  %48="ufront.batchnorm"(%47){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %49="ufront.relu"(%48):(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %50="ufront.conv2d"(%40){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x256x26x26xf32>) -> tensor<1x64x26x26xf32>
  %51="ufront.batchnorm"(%50){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %52="ufront.relu"(%51):(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %53="ufront.conv2d"(%52){groups=1, kernel=[3, 3], pad=[1, 1], stride=[1, 1]}:(tensor<1x64x26x26xf32>) -> tensor<1x96x26x26xf32>
  %54="ufront.batchnorm"(%53){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %55="ufront.relu"(%54):(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %56="ufront.conv2d"(%55){groups=1, kernel=[3, 3], pad=[1, 1], stride=[1, 1]}:(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %57="ufront.batchnorm"(%56){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %58="ufront.relu"(%57):(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %59="ufront.pool2d"(%40){kernel=[3, 3], pad=[1, 1], pool_type="POOL_AVG", stride=[1, 1]}:(tensor<1x256x26x26xf32>) -> tensor<1x256x13x13xf32>
  %60="ufront.conv2d"(%59){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x256x13x13xf32>) -> tensor<1x64x13x13xf32>
  %61="ufront.batchnorm"(%60){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x13x13xf32>) -> tensor<1x64x13x13xf32>
  %62="ufront.relu"(%61):(tensor<1x64x13x13xf32>) -> tensor<1x64x13x13xf32>
  %63="ufront.concat"(%43, %49, %58, %62){axis=1}:(tensor<1x64x26x26xf32>, tensor<1x64x26x26xf32>, tensor<1x96x26x26xf32>, tensor<1x64x13x13xf32>) -> tensor<1x288x26x26xf32>
  %64="ufront.conv2d"(%63){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x288x26x26xf32>) -> tensor<1x64x26x26xf32>
  %65="ufront.batchnorm"(%64){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %66="ufront.relu"(%65):(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %67="ufront.conv2d"(%63){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x288x26x26xf32>) -> tensor<1x48x26x26xf32>
  %68="ufront.batchnorm"(%67){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x48x26x26xf32>) -> tensor<1x48x26x26xf32>
  %69="ufront.relu"(%68):(tensor<1x48x26x26xf32>) -> tensor<1x48x26x26xf32>
  %70="ufront.conv2d"(%69){groups=1, kernel=[5, 5], pad=[2, 2], stride=[1, 1]}:(tensor<1x48x26x26xf32>) -> tensor<1x64x26x26xf32>
  %71="ufront.batchnorm"(%70){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %72="ufront.relu"(%71):(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %73="ufront.conv2d"(%63){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x288x26x26xf32>) -> tensor<1x64x26x26xf32>
  %74="ufront.batchnorm"(%73){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %75="ufront.relu"(%74):(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %76="ufront.conv2d"(%75){groups=1, kernel=[3, 3], pad=[1, 1], stride=[1, 1]}:(tensor<1x64x26x26xf32>) -> tensor<1x96x26x26xf32>
  %77="ufront.batchnorm"(%76){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %78="ufront.relu"(%77):(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %79="ufront.conv2d"(%78){groups=1, kernel=[3, 3], pad=[1, 1], stride=[1, 1]}:(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %80="ufront.batchnorm"(%79){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %81="ufront.relu"(%80):(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %82="ufront.pool2d"(%63){kernel=[3, 3], pad=[1, 1], pool_type="POOL_AVG", stride=[1, 1]}:(tensor<1x288x26x26xf32>) -> tensor<1x288x13x13xf32>
  %83="ufront.conv2d"(%82){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x288x13x13xf32>) -> tensor<1x64x13x13xf32>
  %84="ufront.batchnorm"(%83){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x13x13xf32>) -> tensor<1x64x13x13xf32>
  %85="ufront.relu"(%84):(tensor<1x64x13x13xf32>) -> tensor<1x64x13x13xf32>
  %86="ufront.concat"(%66, %72, %81, %85){axis=1}:(tensor<1x64x26x26xf32>, tensor<1x64x26x26xf32>, tensor<1x96x26x26xf32>, tensor<1x64x13x13xf32>) -> tensor<1x288x26x26xf32>
  %87="ufront.conv2d"(%86){groups=1, kernel=[3, 3], pad=[0, 0], stride=[2, 2]}:(tensor<1x288x26x26xf32>) -> tensor<1x384x12x12xf32>
  %88="ufront.batchnorm"(%87){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x12x12xf32>) -> tensor<1x384x12x12xf32>
  %89="ufront.relu"(%88):(tensor<1x384x12x12xf32>) -> tensor<1x384x12x12xf32>
  %90="ufront.conv2d"(%86){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x288x26x26xf32>) -> tensor<1x64x26x26xf32>
  %91="ufront.batchnorm"(%90){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %92="ufront.relu"(%91):(tensor<1x64x26x26xf32>) -> tensor<1x64x26x26xf32>
  %93="ufront.conv2d"(%92){groups=1, kernel=[3, 3], pad=[1, 1], stride=[1, 1]}:(tensor<1x64x26x26xf32>) -> tensor<1x96x26x26xf32>
  %94="ufront.batchnorm"(%93){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %95="ufront.relu"(%94):(tensor<1x96x26x26xf32>) -> tensor<1x96x26x26xf32>
  %96="ufront.conv2d"(%95){groups=1, kernel=[3, 3], pad=[0, 0], stride=[2, 2]}:(tensor<1x96x26x26xf32>) -> tensor<1x96x12x12xf32>
  %97="ufront.batchnorm"(%96){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x96x12x12xf32>) -> tensor<1x96x12x12xf32>
  %98="ufront.relu"(%97):(tensor<1x96x12x12xf32>) -> tensor<1x96x12x12xf32>
  %99="ufront.pool2d"(%86){kernel=[3, 3], pad=[0, 0], pool_type="POOL_MAX", stride=[2, 2]}:(tensor<1x288x26x26xf32>) -> tensor<1x288x13x13xf32>
  %100="ufront.concat"(%89, %98, %99){axis=1}:(tensor<1x384x12x12xf32>, tensor<1x96x12x12xf32>, tensor<1x288x13x13xf32>) -> tensor<1x768x12x12xf32>
  %101="ufront.conv2d"(%100){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x192x12x12xf32>
  %102="ufront.batchnorm"(%101){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %103="ufront.relu"(%102):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %104="ufront.conv2d"(%100){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x128x12x12xf32>
  %105="ufront.batchnorm"(%104){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %106="ufront.relu"(%105):(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %107="ufront.conv2d"(%106){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %108="ufront.batchnorm"(%107){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %109="ufront.relu"(%108):(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %110="ufront.conv2d"(%109){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x128x12x12xf32>) -> tensor<1x192x12x12xf32>
  %111="ufront.batchnorm"(%110){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %112="ufront.relu"(%111):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %113="ufront.conv2d"(%100){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x128x12x12xf32>
  %114="ufront.batchnorm"(%113){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %115="ufront.relu"(%114):(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %116="ufront.conv2d"(%115){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %117="ufront.batchnorm"(%116){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %118="ufront.relu"(%117):(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %119="ufront.conv2d"(%118){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %120="ufront.batchnorm"(%119){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %121="ufront.relu"(%120):(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %122="ufront.conv2d"(%121){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %123="ufront.batchnorm"(%122){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %124="ufront.relu"(%123):(tensor<1x128x12x12xf32>) -> tensor<1x128x12x12xf32>
  %125="ufront.conv2d"(%124){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x128x12x12xf32>) -> tensor<1x192x12x12xf32>
  %126="ufront.batchnorm"(%125){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %127="ufront.relu"(%126):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %128="ufront.pool2d"(%100){kernel=[3, 3], pad=[1, 1], pool_type="POOL_AVG", stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x768x6x6xf32>
  %129="ufront.conv2d"(%128){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x6x6xf32>) -> tensor<1x192x6x6xf32>
  %130="ufront.batchnorm"(%129){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x6x6xf32>) -> tensor<1x192x6x6xf32>
  %131="ufront.relu"(%130):(tensor<1x192x6x6xf32>) -> tensor<1x192x6x6xf32>
  %132="ufront.concat"(%103, %112, %127, %131){axis=1}:(tensor<1x192x12x12xf32>, tensor<1x192x12x12xf32>, tensor<1x192x12x12xf32>, tensor<1x192x6x6xf32>) -> tensor<1x768x12x12xf32>
  %133="ufront.conv2d"(%132){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x192x12x12xf32>
  %134="ufront.batchnorm"(%133){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %135="ufront.relu"(%134):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %136="ufront.conv2d"(%132){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x160x12x12xf32>
  %137="ufront.batchnorm"(%136){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %138="ufront.relu"(%137):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %139="ufront.conv2d"(%138){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %140="ufront.batchnorm"(%139){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %141="ufront.relu"(%140):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %142="ufront.conv2d"(%141){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x192x12x12xf32>
  %143="ufront.batchnorm"(%142){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %144="ufront.relu"(%143):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %145="ufront.conv2d"(%132){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x160x12x12xf32>
  %146="ufront.batchnorm"(%145){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %147="ufront.relu"(%146):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %148="ufront.conv2d"(%147){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %149="ufront.batchnorm"(%148){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %150="ufront.relu"(%149):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %151="ufront.conv2d"(%150){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %152="ufront.batchnorm"(%151){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %153="ufront.relu"(%152):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %154="ufront.conv2d"(%153){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %155="ufront.batchnorm"(%154){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %156="ufront.relu"(%155):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %157="ufront.conv2d"(%156){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x192x12x12xf32>
  %158="ufront.batchnorm"(%157){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %159="ufront.relu"(%158):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %160="ufront.pool2d"(%132){kernel=[3, 3], pad=[1, 1], pool_type="POOL_AVG", stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x768x6x6xf32>
  %161="ufront.conv2d"(%160){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x6x6xf32>) -> tensor<1x192x6x6xf32>
  %162="ufront.batchnorm"(%161){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x6x6xf32>) -> tensor<1x192x6x6xf32>
  %163="ufront.relu"(%162):(tensor<1x192x6x6xf32>) -> tensor<1x192x6x6xf32>
  %164="ufront.concat"(%135, %144, %159, %163){axis=1}:(tensor<1x192x12x12xf32>, tensor<1x192x12x12xf32>, tensor<1x192x12x12xf32>, tensor<1x192x6x6xf32>) -> tensor<1x768x12x12xf32>
  %165="ufront.conv2d"(%164){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x192x12x12xf32>
  %166="ufront.batchnorm"(%165){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %167="ufront.relu"(%166):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %168="ufront.conv2d"(%164){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x160x12x12xf32>
  %169="ufront.batchnorm"(%168){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %170="ufront.relu"(%169):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %171="ufront.conv2d"(%170){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %172="ufront.batchnorm"(%171){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %173="ufront.relu"(%172):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %174="ufront.conv2d"(%173){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x192x12x12xf32>
  %175="ufront.batchnorm"(%174){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %176="ufront.relu"(%175):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %177="ufront.conv2d"(%164){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x160x12x12xf32>
  %178="ufront.batchnorm"(%177){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %179="ufront.relu"(%178):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %180="ufront.conv2d"(%179){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %181="ufront.batchnorm"(%180){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %182="ufront.relu"(%181):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %183="ufront.conv2d"(%182){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %184="ufront.batchnorm"(%183){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %185="ufront.relu"(%184):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %186="ufront.conv2d"(%185){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %187="ufront.batchnorm"(%186){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %188="ufront.relu"(%187):(tensor<1x160x12x12xf32>) -> tensor<1x160x12x12xf32>
  %189="ufront.conv2d"(%188){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x160x12x12xf32>) -> tensor<1x192x12x12xf32>
  %190="ufront.batchnorm"(%189){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %191="ufront.relu"(%190):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %192="ufront.pool2d"(%164){kernel=[3, 3], pad=[1, 1], pool_type="POOL_AVG", stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x768x6x6xf32>
  %193="ufront.conv2d"(%192){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x6x6xf32>) -> tensor<1x192x6x6xf32>
  %194="ufront.batchnorm"(%193){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x6x6xf32>) -> tensor<1x192x6x6xf32>
  %195="ufront.relu"(%194):(tensor<1x192x6x6xf32>) -> tensor<1x192x6x6xf32>
  %196="ufront.concat"(%167, %176, %191, %195){axis=1}:(tensor<1x192x12x12xf32>, tensor<1x192x12x12xf32>, tensor<1x192x12x12xf32>, tensor<1x192x6x6xf32>) -> tensor<1x768x12x12xf32>
  %197="ufront.conv2d"(%196){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x192x12x12xf32>
  %198="ufront.batchnorm"(%197){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %199="ufront.relu"(%198):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %200="ufront.conv2d"(%196){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x192x12x12xf32>
  %201="ufront.batchnorm"(%200){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %202="ufront.relu"(%201):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %203="ufront.conv2d"(%202){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %204="ufront.batchnorm"(%203){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %205="ufront.relu"(%204):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %206="ufront.conv2d"(%205){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %207="ufront.batchnorm"(%206){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %208="ufront.relu"(%207):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %209="ufront.conv2d"(%196){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x192x12x12xf32>
  %210="ufront.batchnorm"(%209){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %211="ufront.relu"(%210):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %212="ufront.conv2d"(%211){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %213="ufront.batchnorm"(%212){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %214="ufront.relu"(%213):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %215="ufront.conv2d"(%214){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %216="ufront.batchnorm"(%215){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %217="ufront.relu"(%216):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %218="ufront.conv2d"(%217){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %219="ufront.batchnorm"(%218){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %220="ufront.relu"(%219):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %221="ufront.conv2d"(%220){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %222="ufront.batchnorm"(%221){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %223="ufront.relu"(%222):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %224="ufront.pool2d"(%196){kernel=[3, 3], pad=[1, 1], pool_type="POOL_AVG", stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x768x6x6xf32>
  %225="ufront.conv2d"(%224){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x6x6xf32>) -> tensor<1x192x6x6xf32>
  %226="ufront.batchnorm"(%225){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x6x6xf32>) -> tensor<1x192x6x6xf32>
  %227="ufront.relu"(%226):(tensor<1x192x6x6xf32>) -> tensor<1x192x6x6xf32>
  %228="ufront.concat"(%199, %208, %223, %227){axis=1}:(tensor<1x192x12x12xf32>, tensor<1x192x12x12xf32>, tensor<1x192x12x12xf32>, tensor<1x192x6x6xf32>) -> tensor<1x768x12x12xf32>
  %229="ufront.conv2d"(%228){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x192x12x12xf32>
  %230="ufront.batchnorm"(%229){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %231="ufront.relu"(%230):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %232="ufront.conv2d"(%231){groups=1, kernel=[3, 3], pad=[0, 0], stride=[2, 2]}:(tensor<1x192x12x12xf32>) -> tensor<1x320x5x5xf32>
  %233="ufront.batchnorm"(%232){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x320x5x5xf32>) -> tensor<1x320x5x5xf32>
  %234="ufront.relu"(%233):(tensor<1x320x5x5xf32>) -> tensor<1x320x5x5xf32>
  %235="ufront.conv2d"(%228){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x768x12x12xf32>) -> tensor<1x192x12x12xf32>
  %236="ufront.batchnorm"(%235){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %237="ufront.relu"(%236):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %238="ufront.conv2d"(%237){groups=1, kernel=[1, 7], pad=[0, 3], stride=[1, 1]}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %239="ufront.batchnorm"(%238){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %240="ufront.relu"(%239):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %241="ufront.conv2d"(%240){groups=1, kernel=[7, 1], pad=[3, 0], stride=[1, 1]}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %242="ufront.batchnorm"(%241){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %243="ufront.relu"(%242):(tensor<1x192x12x12xf32>) -> tensor<1x192x12x12xf32>
  %244="ufront.conv2d"(%243){groups=1, kernel=[3, 3], pad=[0, 0], stride=[2, 2]}:(tensor<1x192x12x12xf32>) -> tensor<1x192x5x5xf32>
  %245="ufront.batchnorm"(%244){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x5x5xf32>) -> tensor<1x192x5x5xf32>
  %246="ufront.relu"(%245):(tensor<1x192x5x5xf32>) -> tensor<1x192x5x5xf32>
  %247="ufront.pool2d"(%228){kernel=[3, 3], pad=[0, 0], pool_type="POOL_MAX", stride=[2, 2]}:(tensor<1x768x12x12xf32>) -> tensor<1x768x6x6xf32>
  %248="ufront.concat"(%234, %246, %247){axis=1}:(tensor<1x320x5x5xf32>, tensor<1x192x5x5xf32>, tensor<1x768x6x6xf32>) -> tensor<1x1280x5x5xf32>
  %249="ufront.conv2d"(%248){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x1280x5x5xf32>) -> tensor<1x320x5x5xf32>
  %250="ufront.batchnorm"(%249){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x320x5x5xf32>) -> tensor<1x320x5x5xf32>
  %251="ufront.relu"(%250):(tensor<1x320x5x5xf32>) -> tensor<1x320x5x5xf32>
  %252="ufront.conv2d"(%248){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x1280x5x5xf32>) -> tensor<1x384x5x5xf32>
  %253="ufront.batchnorm"(%252){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %254="ufront.relu"(%253):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %255="ufront.conv2d"(%254){groups=1, kernel=[1, 3], pad=[0, 1], stride=[1, 1]}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %256="ufront.batchnorm"(%255){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %257="ufront.relu"(%256):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %258="ufront.conv2d"(%254){groups=1, kernel=[3, 1], pad=[1, 0], stride=[1, 1]}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %259="ufront.batchnorm"(%258){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %260="ufront.relu"(%259):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %261="ufront.concat"(%257, %260){axis=1}:(tensor<1x384x5x5xf32>, tensor<1x384x5x5xf32>) -> tensor<1x768x5x5xf32>
  %262="ufront.conv2d"(%248){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x1280x5x5xf32>) -> tensor<1x448x5x5xf32>
  %263="ufront.batchnorm"(%262){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x448x5x5xf32>) -> tensor<1x448x5x5xf32>
  %264="ufront.relu"(%263):(tensor<1x448x5x5xf32>) -> tensor<1x448x5x5xf32>
  %265="ufront.conv2d"(%264){groups=1, kernel=[3, 3], pad=[1, 1], stride=[1, 1]}:(tensor<1x448x5x5xf32>) -> tensor<1x384x5x5xf32>
  %266="ufront.batchnorm"(%265){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %267="ufront.relu"(%266):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %268="ufront.conv2d"(%267){groups=1, kernel=[1, 3], pad=[0, 1], stride=[1, 1]}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %269="ufront.batchnorm"(%268){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %270="ufront.relu"(%269):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %271="ufront.conv2d"(%267){groups=1, kernel=[3, 1], pad=[1, 0], stride=[1, 1]}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %272="ufront.batchnorm"(%271){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %273="ufront.relu"(%272):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %274="ufront.concat"(%270, %273){axis=1}:(tensor<1x384x5x5xf32>, tensor<1x384x5x5xf32>) -> tensor<1x768x5x5xf32>
  %275="ufront.pool2d"(%248){kernel=[3, 3], pad=[1, 1], pool_type="POOL_AVG", stride=[1, 1]}:(tensor<1x1280x5x5xf32>) -> tensor<1x1280x2x2xf32>
  %276="ufront.conv2d"(%275){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x1280x2x2xf32>) -> tensor<1x192x2x2xf32>
  %277="ufront.batchnorm"(%276){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x2x2xf32>) -> tensor<1x192x2x2xf32>
  %278="ufront.relu"(%277):(tensor<1x192x2x2xf32>) -> tensor<1x192x2x2xf32>
  %279="ufront.concat"(%251, %261, %274, %278){axis=1}:(tensor<1x320x5x5xf32>, tensor<1x768x5x5xf32>, tensor<1x768x5x5xf32>, tensor<1x192x2x2xf32>) -> tensor<1x2048x5x5xf32>
  %280="ufront.conv2d"(%279){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x2048x5x5xf32>) -> tensor<1x320x5x5xf32>
  %281="ufront.batchnorm"(%280){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x320x5x5xf32>) -> tensor<1x320x5x5xf32>
  %282="ufront.relu"(%281):(tensor<1x320x5x5xf32>) -> tensor<1x320x5x5xf32>
  %283="ufront.conv2d"(%279){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x2048x5x5xf32>) -> tensor<1x384x5x5xf32>
  %284="ufront.batchnorm"(%283){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %285="ufront.relu"(%284):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %286="ufront.conv2d"(%285){groups=1, kernel=[1, 3], pad=[0, 1], stride=[1, 1]}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %287="ufront.batchnorm"(%286){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %288="ufront.relu"(%287):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %289="ufront.conv2d"(%285){groups=1, kernel=[3, 1], pad=[1, 0], stride=[1, 1]}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %290="ufront.batchnorm"(%289){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %291="ufront.relu"(%290):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %292="ufront.concat"(%288, %291){axis=1}:(tensor<1x384x5x5xf32>, tensor<1x384x5x5xf32>) -> tensor<1x768x5x5xf32>
  %293="ufront.conv2d"(%279){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x2048x5x5xf32>) -> tensor<1x448x5x5xf32>
  %294="ufront.batchnorm"(%293){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x448x5x5xf32>) -> tensor<1x448x5x5xf32>
  %295="ufront.relu"(%294):(tensor<1x448x5x5xf32>) -> tensor<1x448x5x5xf32>
  %296="ufront.conv2d"(%295){groups=1, kernel=[3, 3], pad=[1, 1], stride=[1, 1]}:(tensor<1x448x5x5xf32>) -> tensor<1x384x5x5xf32>
  %297="ufront.batchnorm"(%296){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %298="ufront.relu"(%297):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %299="ufront.conv2d"(%298){groups=1, kernel=[1, 3], pad=[0, 1], stride=[1, 1]}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %300="ufront.batchnorm"(%299){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %301="ufront.relu"(%300):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %302="ufront.conv2d"(%298){groups=1, kernel=[3, 1], pad=[1, 0], stride=[1, 1]}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %303="ufront.batchnorm"(%302){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %304="ufront.relu"(%303):(tensor<1x384x5x5xf32>) -> tensor<1x384x5x5xf32>
  %305="ufront.concat"(%301, %304){axis=1}:(tensor<1x384x5x5xf32>, tensor<1x384x5x5xf32>) -> tensor<1x768x5x5xf32>
  %306="ufront.pool2d"(%279){kernel=[3, 3], pad=[1, 1], pool_type="POOL_AVG", stride=[1, 1]}:(tensor<1x2048x5x5xf32>) -> tensor<1x2048x2x2xf32>
  %307="ufront.conv2d"(%306){groups=1, kernel=[1, 1], pad=[0, 0], stride=[1, 1]}:(tensor<1x2048x2x2xf32>) -> tensor<1x192x2x2xf32>
  %308="ufront.batchnorm"(%307){affine=true, eps=0.001, momentum=0.1, track_running_stats=true}:(tensor<1x192x2x2xf32>) -> tensor<1x192x2x2xf32>
  %309="ufront.relu"(%308):(tensor<1x192x2x2xf32>) -> tensor<1x192x2x2xf32>
  %310="ufront.concat"(%282, %292, %305, %309){axis=1}:(tensor<1x320x5x5xf32>, tensor<1x768x5x5xf32>, tensor<1x768x5x5xf32>, tensor<1x192x2x2xf32>) -> tensor<1x2048x5x5xf32>
  %311="ufront.pool2d"(%310){output_size=[1, 1], pool_type="POOL_ADAPTIVE"}:(tensor<1x2048x5x5xf32>) -> tensor<1x2048x1x1xf32>
  %312="ufront.dropout"(%311){rate=0.5, seed=0}:(tensor<1x2048x1x1xf32>) -> tensor<1x2048x1x1xf32>
  %313="ufront.flat"(%312){end_dim=-1, start_dim=1}:(tensor<1x2048x1x1xf32>) -> tensor<1x2048xf32>
  %314="ufront.linear"(%313):(tensor<1x2048xf32>) -> tensor<1x1000xf32>
  return %314: tensor<1x1000xf32>
}